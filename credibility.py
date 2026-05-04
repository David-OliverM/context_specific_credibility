#!/usr/bin/env python
import os 
import sys
from packages import PACKAGE_DICT 
for package in PACKAGE_DICT:
    print(f"Adding package: {package} to sys.path. Given path: {os.path.join('packages', PACKAGE_DICT[package])}")
    sys.path.append(os.path.join("packages", PACKAGE_DICT[package]))
import omegaconf
import time
import wandb
from hydra.core.hydra_config import HydraConfig
import logging
from omegaconf import DictConfig, OmegaConf, open_dict
import os
from rich.traceback import install
install()
import hydra
import pytorch_lightning as pl
import torch.utils.data
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import StochasticWeightAveraging, RichProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import (
    ModelSummary,
)
from utils import (
    load_from_checkpoint,
)

from datasets import get_dataloader
from models.base import LateFusionClassifier, LateFusionMultiLabelClassifier, FusionModel
from models import *
import json 
from models.save_results import *

# A logger for this file
logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)
DATALOADER_ID_TO_SET_NAME = {0: "train", 1: "val", 2: "test"}
 
class JSD(torch.nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = torch.nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor, eps=1e-12):
        p, q = p.clamp(eps), q.clamp(eps)
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(p.log(), m) + self.kl(q.log(), m))
    
class CSLateFusionClassifier(LateFusionClassifier):
    """
    Noisy Late fusion model. Outputs the probability distribution over a target variable given input from multiple modalities
    """
    def __init__(self, cfg: DictConfig, dataloaders: list, steps_per_epoch: int, name="NoisyLateFusionClassifier"):
        super().__init__(cfg, name=name, steps_per_epoch=steps_per_epoch)
        self.test_corruption = []
        self.test_credibility = []
        self.corrupt_modalities = []
        self.credible_modality = []

    def _get_cross_entropy_and_accuracy(self, batch, noise_ind) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross entropy loss and accuracy of batch.
        Args:
            batch: Batch of data.

        Returns:
            Tuple of (cross entropy loss, accuracy).
        """
        # Sanity check that there are #modalities + 1(target) variables in input

        assert len(batch) == self.num_modalities + 1
        
        data, labels = batch[:-1], batch[-1]
        loss, embeddings, predictions = 0.0,  [], []
        noise_encoders = self.noise_encoders
        corruptions = []
        if self.cfg.random_noise_context:
            noise_ind = [torch.rand_like(noise) for noise in noise_ind]
        for unimodal_data, encoder, noise_encoder, predictor,  noise in zip(data, self.encoders, noise_encoders, self.predictors,  noise_ind):
            if self.cfg.fully_decoupled_training:
                with torch.no_grad():
                    embeddings += [encoder(unimodal_data)]
                    corruptions += [noise_encoder(noise)]
                    unimodal_prediction = predictor(embeddings[-1])
            
            else:
                if(encoder is not None):
                    embeddings += [encoder(unimodal_data)]
                else:
                    embeddings += [unimodal_data]

                corruptions += [noise_encoder(noise)]    
                unimodal_prediction = predictor(embeddings[-1])
            

            if(self.cfg.experiment.head.threshold_input):
                predictions += [unimodal_prediction.argmax(dim=-1).unsqueeze(1)]
            else:
                predictions += [unimodal_prediction.unsqueeze(1)]
            ll_y_g_x = unimodal_prediction.log()
            loss += self.criterion(ll_y_g_x, labels) if not self.cfg.fully_decoupled_training else 0

        predictions = torch.cat(predictions, dim=1)
        unimodal_predictions = torch.permute(predictions, (1,0,2))
        context = torch.cat(embeddings, dim=-1)
        # if (hasattr(self.cfg.experiment.head, "noise_context") and not self.cfg.experiment.head.noise_context):
        #     context = torch.cat(embeddings, dim=-1)

        # else:
        #     context = [torch.cat((t1, t2), dim=-1) for t1, t2 in zip(embeddings, corruptions)]
        #     context = torch.cat(context, dim=-1)



        if(not self.cfg.joint_training or self.current_epoch <=5):
            predictions = predictions.detach()
            context = context.detach()
        
        predictions_in = predictions.detach()
        context = context.detach()
        predictions = self.head(predictions_in, context)

        # Criterion is NLL which takes logp( y | x)
        # NOTE: Don't use nn.CrossEntropyLoss because it expects unnormalized logits
        # and applies LogSoftmax first
        ll_y_g_x = predictions.log()
        
        loss += self.criterion(ll_y_g_x, labels)
        if(hasattr(self.head,"loss")):
            loss += self.head.loss[:,labels].mean()
        accuracy = (labels == ll_y_g_x.argmax(-1)).sum() / ll_y_g_x.shape[0]
        return loss, accuracy, predictions_in, predictions, context
    
    

        
    
    def training_step(self, train_batch, batch_idx):
        train_batch,  _, noise_ind = train_batch
        loss, accuracy, predictions_in, predictions_out, context = self._get_cross_entropy_and_accuracy(train_batch, noise_ind=noise_ind)
        self.log("Train/accuracy", accuracy, on_step=True, prog_bar=True)
        self.log("Train/loss", loss, on_step=True)
        self.log_metrics(predictions_in, predictions_out, train_batch[-1], "Train/", on_step=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        val_batch, _, noise_ind = val_batch
        loss, accuracy, predictions_in, predictions_out, context = self._get_cross_entropy_and_accuracy(val_batch, noise_ind=noise_ind)
        self.log("Val/accuracy", accuracy, prog_bar=True)
        self.log("Val/loss", loss)
        self.log_metrics(predictions_in, predictions_out, val_batch[-1], "Val/")
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=2):
        batch, (index, sample_corruption, corr_modalities), noise_ind = batch
        corruption = [corr1 if corr1 != 'none' else corr2 for (corr1, corr2) in zip(sample_corruption[0], sample_corruption[1])]
        loss, accuracy, predictions_in, predictions_out, context = self._get_cross_entropy_and_accuracy(batch, noise_ind=noise_ind)
        set_name = DATALOADER_ID_TO_SET_NAME[dataloader_idx]        
        if(set_name == "test"):
            self.test_pred += [predictions_out]
            self.test_target += [batch[-1]]
            self.test_corruption.extend(corruption)
            self.corrupt_modalities+= [corr_modalities.to(int)]
            self.unimodal_test_pred += [predictions_in]
        self.log(f"Test/{set_name}_accuracy", accuracy, add_dataloader_idx=False)
        self.log_metrics(predictions_in, predictions_out, context, batch[-1], f"Test/{set_name}_", add_dataloader_idx=False)

    def on_test_end(self) -> None:
        self.test_pred   = torch.cat(self.test_pred).cpu().detach()
        self.test_target = torch.cat(self.test_target).cpu().detach()
        self.test_credibility = torch.cat(self.test_credibility).cpu().detach()
        self.corrupt_modalities = torch.cat(self.corrupt_modalities).cpu().detach()
        self.credible_modality = self.test_credibility.argmax(dim=-1).cpu().detach()
        noisy_sample_indices = torch.nonzero(self.corrupt_modalities.sum(dim=1) > 0).squeeze()
        test_cred_global = self.test_credibility.mean(dim=0)
        cred_ind = [1 if self.corrupt_modalities[i].sum()==0 else (self.corrupt_modalities[i].argmin() == self.credible_modality[i]).to(int).item() for i in range(self.corrupt_modalities.shape[0])]
        cred_ind = torch.tensor(cred_ind)
        global_cred_ind = cred_ind.sum()/cred_ind.shape[0]
        print(f"Credibility: \nModality 1: {test_cred_global[0]}\nModality 2: {test_cred_global[1]}")
        for metric_name in self.metrics:
            metric = self.metrics[metric_name].cpu()
            score = metric(self.test_pred, self.test_target)
            print("Test/", metric_name, score)
        
        print("Class wise credibility scores:")
        cred_ind_for_corr_samples = {}
        for cls in range(self.cfg.experiment.dataset.num_classes):
            cls_indices = (self.test_target == cls).nonzero(as_tuple=True)[0]
            test_cred_by_cls = self.test_credibility[cls_indices]
            avg_credibility_by_cls = test_cred_by_cls.mean(dim=0)
            print(f"Class {cls}: Modality 1: {avg_credibility_by_cls[0]} Modality 2: {avg_credibility_by_cls[1]}")

            cred_ind_by_cls = cred_ind[cls_indices]
            cred_ind_by_cls = cred_ind_by_cls.sum()/cred_ind_by_cls.shape[0]
            noisy_samples_by_cls = self.corrupt_modalities[cls_indices]
            noisy_sample_indices_by_cls = torch.nonzero(noisy_samples_by_cls.sum(dim=1) > 0).squeeze()
            cred_ind_for_corr_samples_by_cls = cred_ind[cls_indices][noisy_sample_indices_by_cls]
            cred_ind_corr_samples_by_cls = 0 if noisy_sample_indices_by_cls.shape[0] == 0 else (cred_ind_for_corr_samples_by_cls.sum()/noisy_sample_indices_by_cls.shape[0]).item()
            cred_ind_for_corr_samples[str(cls)] = cred_ind_corr_samples_by_cls
        test_noise = self.cfg.test_noise * 100
        save_experiment_metrics(
                test_pred=self.test_pred,
                test_target=self.test_target,
                test_credibility=self.test_credibility,
                metrics=self.metrics,
                cred_ind = cred_ind,
                cred_ind_corr_samples = cred_ind_for_corr_samples,
                group_tag=self.cfg.group_tag,
                experiment_name=self.cfg.experiment.name,
                exp_setup=self.cfg.exp_setup,
                noise_setting=self.cfg.noise_severity,
                test_noise=test_noise,
                run_id=self.cfg.seed,
                num_classes = self.cfg.experiment.dataset.num_classes,
                save_dir = f"/nas1-nfs1/home/pxt220000/projects/CS_Credibility/new/scores/{self.cfg.experiment.dataset.name}",
                filename=f"results_test_noise_{test_noise}.json"
            ) 

        




    
    def log_metrics(self, predictions_in, predictions_out, context, targets, mode='Train/', **kwargs):
        credibility = []
        probs = self.head.model(predictions_in.unsqueeze(1).unsqueeze(3).unsqueeze(3),cond_input=context,marginalized_scopes=None).exp()
        probs = probs/probs.sum(dim=-1,keepdim=True)
        for i in range(self.cfg.experiment.dataset.modalities):
            p_y_pi = self.head.model(predictions_in.unsqueeze(1).unsqueeze(3).unsqueeze(3),cond_input=context,marginalized_scopes=[i]).exp()
            p_y_pi = p_y_pi/p_y_pi.sum(dim=-1, keepdim=True)
            credibility += [torch.nn.functional.kl_div(probs.log(), p_y_pi, reduction='none').sum(dim=-1, keepdim=True)]
        credibility = torch.cat(credibility,dim=-1)
        credibility = credibility/credibility.sum(dim=-1, keepdim=True)
        self.test_credibility += [credibility]
        batch_credibility = credibility.mean(dim=0).detach().cpu()
        for i in range(self.cfg.experiment.dataset.modalities):     
            self.log(f"{mode}Credibility-Modality-{i}",batch_credibility[i])
        probs, targets = predictions_out.detach().cpu(), targets.detach().cpu()
        for metric_name in self.metrics:
            self.log(f"{mode}{metric_name}",self.metrics[metric_name](probs, targets),**kwargs)
        

def main(cfg: DictConfig):
    """
    Main function for training and evaluating an Einet.

    Args:
        cfg: Config file.
    """
    preprocess_cfg(cfg)

    # Get hydra config
    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.runtime.output_dir
    logger.info("Working directory : {}".format(os.getcwd()))

    # Save config
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f)

    # Safe run_dir in config (use open_dict to make config writable)
    with open_dict(cfg):
        cfg.run_dir = run_dir

    logger.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))
    logger.info("Run dir: " + run_dir)

    seed_everything(cfg.seed, workers=True)

    if not cfg.wandb:
        os.environ["WANDB_MODE"] = "offline"

    # Ensure that everything is properly seeded
    seed_everything(cfg.seed, workers=True)

    # Setup devices
    if torch.cuda.is_available():
        accelerator = "gpu"
        if type(cfg.gpu) == int:
            devices = [int(cfg.gpu)]
        else:
            devices = [int(g) for g in cfg.gpu]
    else:
        accelerator = "cpu"
        devices = 1

    # Create dataloader
    train_loader, val_loader, test_loader = get_dataloader(cfg)

    # Create callbacks
    cfg_container = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger_wandb = WandbLogger(
        name=cfg.tag,
        project=cfg.project_name,
        group=cfg.group_tag,
        offline=not cfg.wandb,
        config=cfg_container,
        reinit=True,
        save_dir=run_dir,
        settings=wandb.Settings(start_method="thread"),
    )

    # Load or create model
    base_model_class = LateFusionMultiLabelClassifier if(cfg.experiment.multilabel) else LateFusionClassifier
    noisy_model_class = None if(cfg.experiment.multilabel) else CSLateFusionClassifier
    base_model = base_model_class.load_from_checkpoint(f"{run_dir}/best_model.pt")
    noisy_model = noisy_model_class(cfg, [train_loader,val_loader, test_loader], steps_per_epoch=len(train_loader))
    noisy_model.load_state_dict(base_model.state_dict())
    
    # Setup callbacks
    ckpt_callback = ModelCheckpoint(f"{run_dir}/checkpoints", monitor="Val/F1Score", save_top_k=3, mode='max')
    callbacks = [
        ckpt_callback
    ]
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        logger=logger_wandb,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        precision=cfg.precision,
        fast_dev_run=cfg.debug,
        profiler=cfg.profiler,
        default_root_dir=run_dir,
        enable_checkpointing=True,
        detect_anomaly=True,
    )


    # performance_summary = {
    #     "no_noise": {},
    #     "noise_in_test": {},
    #     "noise_in_test_and_train":{}
    # }
    
            
    logger.info("Evaluating the Model...")
    trainer.test(model=noisy_model, dataloaders=[test_loader], verbose=True)
    
        
    # for metric_name in noisy_model.metrics:
    #     metric = noisy_model.metrics[metric_name].cpu()
    #     performance_summary["noise_in_test"][metric_name] = metric(noisy_model.test_pred, noisy_model.test_target).item()
        
        
    # print(performance_summary)
    # summary_dir = os.path.join(run_dir, "summary")
    # os.makedirs(summary_dir, exist_ok=True)
    # with open(os.path.join(summary_dir,'credibility_scores.json'), 'w') as fp:
    #     json.dump(performance_summary, fp)
    
                
def preprocess_cfg(cfg: DictConfig):
    """
    Preprocesses the config file.
    Replace defaults if not set (such as data/results dir).

    Args:
        cfg: Config file.
    """
    home = os.getenv("HOME")
    
    # If FP16/FP32 is given, convert to int (else it's "bf16", keep string)
    if cfg.precision == "16" or cfg.precision == "32":
        cfg.precision = int(cfg.precision)

    if "profiler" not in cfg:
        cfg.profiler = None  # Accepted by PyTorch Lightning Trainer class

    if "tag" not in cfg:
        cfg.tag = cfg.experiment.name+"-noisy"

    # cfg.group_tag = "credibility-analysis"

    if "seed" not in cfg:
        cfg.seed = int(time.time())
        

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main_hydra(cfg: DictConfig):
    try:
        main(cfg)
    except Exception as e:
        logging.critical(e, exc_info=True)  # log exception info at CRITICAL log level
    finally:
        # Close wandb instance. Necessary for hydra multi-runs where main() is called multipel times
        wandb.finish()


if __name__ == "__main__":
    main_hydra()