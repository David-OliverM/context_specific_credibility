from abc import ABC
from typing import Tuple
import pytorch_lightning as pl
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision
from omegaconf import DictConfig
from rtpt import RTPT
from torch import nn
from models import *
import torchmetrics
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
from sklearn.calibration import calibration_curve
from packages.probmetrics.probmetrics.distributions import CategoricalProbs
from packages.probmetrics.probmetrics.calibrators import get_calibrator
from omegaconf import open_dict
from models.reliability_diagrams import *
from models.losses import FLandMDCA, SupConLoss
from models.save_results import save_metric_json
# Translate the dataloader index to the dataset name
DATALOADER_ID_TO_SET_NAME = {0: "train", 1: "val", 2: "test"}

class LitModel(pl.LightningModule, ABC):
    """
    LightningModule for training a model using PyTorch Lightning.

    Args:
        cfg (DictConfig): Configuration dictionary.
        name (str): Name of the model.
        steps_per_epoch (int): Number of steps per epoch.

    Attributes:
        cfg (DictConfig): Configuration dictionary.
        image_shape (ImageShape): Shape of the input data.
        rtpt (RTPT): RTPT logger.
        steps_per_epoch (int): Number of steps per epoch.
    """

    def __init__(self, cfg: DictConfig, name: str, steps_per_epoch: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.rtpt = RTPT(
            name_initials="MultiModal",
            experiment_name="fusion_" + name + ("_" + str(cfg.tag) if cfg.tag else ""),
            max_iterations=cfg.epochs + 1,
        )
        self.save_hyperparameters()
        self.steps_per_epoch = steps_per_epoch
        self.configure_metrics()
    
    def configure_metrics(self):
       return "Not Implemented"
   
    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(0.7 * self.cfg.epochs), int(0.9 * self.cfg.epochs)],
            gamma=0.1,
        )
        return [optimizer], [lr_scheduler]

    def on_train_start(self) -> None:
        self.rtpt.start()

    def on_train_epoch_end(self) -> None:
        self.rtpt.step()



class FusionModel(LitModel):
    """
    Fusion model. Outputs the probability distribution over a target variable given input from multiple modalities
    """

    def __init__(self, cfg: DictConfig, steps_per_epoch: int, name="LateFusion"):
        super().__init__(cfg, name=name, steps_per_epoch=steps_per_epoch)
        self.cfg = cfg
        self.encoders, self.predictors = [], []
        for modality in cfg.experiment.encoders:
            if(cfg.experiment.encoders[modality].type is None):
                self.encoders += [None]
            else:
                self.encoders += [eval(cfg.experiment.encoders[modality].type)(**cfg.experiment.encoders[modality].args, freeze_params = cfg.freeze_unimodals)]
        for modality in cfg.experiment.predictors:
            self.predictors += [eval(cfg.experiment.predictors[modality].type)(**cfg.experiment.predictors[modality].args)] if cfg.experiment.predictors[modality].type is not None else [None]

        self.encoders, self.predictors = torch.nn.ModuleList(self.encoders), torch.nn.ModuleList(self.predictors)
        self.num_modalities = cfg.experiment.dataset.modalities
        if self.cfg.fully_decoupled_training:
            encoders, predictors = [], []
            for i in range(self.num_modalities):
                unimodal_model = torch.nn.Sequential(self.encoders[i], self.predictors[i])
                unimodal_model.load_state_dict(torch.load(f'trained_models/{self.cfg.experiment.dataset.name}/noise_{self.cfg.noise_severity}/seed_{self.cfg.seed}/unimodal_{i}.pth'))
                encoder = unimodal_model[0]
                predictor = unimodal_model[1]
                encoder.eval()
                predictor.eval()
                for p in encoder.parameters():
                    p.requires_grad = False
                for p in predictor.parameters():
                    p.requires_grad = False
                
                encoders.append(encoder)
                predictors.append(predictor)
            self.encoders, self.predictors = torch.nn.ModuleList(encoders), torch.nn.ModuleList(predictors)

        self.head = eval(cfg.experiment.head.type)(**cfg.experiment.head.args)
        
        # Define loss function
        self.criterion = nn.NLLLoss()
        self.test_pred, self.test_target, self.unimodal_test_pred = [], [], []
        self.val_pred, self.val_target, self.unimodal_val_pred = [], [], []
        self.unimodal_criterion = FLandMDCA(gamma=2.0, beta=0.3) if hasattr(cfg.experiment, "unimodal_criterion") and cfg.experiment.unimodal_criterion=="FLandMDCA" else torch.nn.NLLLoss()
        self.noise_encoders = torch.nn.ModuleList()
        for enc in self.encoders:
            if enc is not None:
                enc_copy = copy.deepcopy(enc)  
                enc_copy.eval()               
                for param in enc_copy.parameters():
                    param.requires_grad = False  # freeze parameters
                self.noise_encoders.append(enc_copy)

    # --------------------------------------------------------------
    # F1.4: per-fold TabPFN setup for TabPFNSAXEncoder modules.
    # --------------------------------------------------------------
    def fit_tabpfn_encoders(self, train_loader, val_loader, test_loader):
        """Walk dataloaders, fit each TabPFNSAXEncoder on the fold's train set,
        then precompute per-sample probabilities for the full fold.

        No-op if there are no TabPFNSAXEncoder instances in self.encoders.
        Encoders that crash during fit (e.g. ``dopamine/other`` with 1792
        SAX features vs TabPFN's 500-feature soft limit) get a uniform
        probability fallback (1/n_classes) so the training run continues
        instead of bringing down the whole pipeline.

        Args:
          train_loader, val_loader, test_loader: the three loaders produced
            by ``get_dataloader(...)``. Their datasets must emit per-modality
            time series ``(T, k_rois)`` -- i.e. the configs must set
            ``experiment.dataset.args.emit_timeseries=true``.
        """
        from models.predictor import TabPFNSAXEncoder
        import numpy as np

        tabpfn_encoders = [
            (i, enc) for i, enc in enumerate(self.encoders)
            if isinstance(enc, TabPFNSAXEncoder)
        ]
        if not tabpfn_encoders:
            return  # No-op for MLPEncoder-only configs.

        print(f"[F1.4] fit_tabpfn_encoders: "
              f"found {len(tabpfn_encoders)} TabPFNSAXEncoder(s); "
              f"setting up per-fold demonstrations.")

        # Walk the train loader and collect (X_per_modality, y, sample_indices).
        def _collect(loader, set_name):
            xs_per_mod = None
            ys, idxs = [], []
            for batch in loader:
                batch_data, (idx_t, _, _), _ = batch
                # batch_data = (mod_1, mod_2, ..., mod_M, label)
                mods = batch_data[:-1]
                labels = batch_data[-1]
                if xs_per_mod is None:
                    xs_per_mod = [[] for _ in mods]
                for m, t in enumerate(mods):
                    xs_per_mod[m].append(t.detach().cpu().numpy())
                ys.append(labels.detach().cpu().numpy())
                # idx_t is a list[int] or a tensor depending on collate behaviour.
                if hasattr(idx_t, "tolist"):
                    idxs.extend(list(idx_t.tolist()))
                else:
                    idxs.extend(list(idx_t))
            xs_per_mod = [np.concatenate(x, axis=0) for x in xs_per_mod]
            y = np.concatenate(ys, axis=0)
            print(f"[F1.4]   {set_name}: n={len(y)}  "
                  f"per-modality shapes={[x.shape for x in xs_per_mod]}")
            return xs_per_mod, y, idxs

        Xtr, ytr, idx_tr = _collect(train_loader, "train")
        Xva, yva, idx_va = _collect(val_loader,   "val  ")
        Xte, yte, idx_te = _collect(test_loader,  "test ")

        # Per-encoder fit + cache precompute. Try/except around fit so a
        # 1792-feature bucket failure does not kill the whole run.
        for enc_idx, enc in tabpfn_encoders:
            m_name = (list(self.cfg.experiment.encoders.keys())[enc_idx]
                      if hasattr(self.cfg.experiment, "encoders") else f"enc{enc_idx}")
            sax_feat_dim = len(enc.roi_indices) * enc.sax_word_size
            print(f"[F1.4]   modality {m_name} (encoder idx {enc_idx}): "
                  f"k_rois={len(enc.roi_indices)}  sax_features={sax_feat_dim}")
            try:
                enc.fit_tabpfn(Xtr[enc_idx], ytr)
                # Combine train+val+test for the precompute cache so forward()
                # can look up any sample seen during fit/val/test loops.
                X_all = np.concatenate([Xtr[enc_idx], Xva[enc_idx], Xte[enc_idx]], axis=0)
                idx_all = list(idx_tr) + list(idx_va) + list(idx_te)
                enc.precompute_probs(X_all, idx_all)
                print(f"[F1.4]     fit OK; probs cache size = "
                      f"{len(enc._probs_cache)}")
            except Exception as e:  # pragma: no cover - documented fallback
                print(f"[F1.4]   WARNING: TabPFN fit/precompute failed for "
                      f"{m_name}: {type(e).__name__}: {e}")
                print(f"[F1.4]   Falling back to uniform-probability mode for "
                      f"this encoder.  Run will continue.")
                # We don't substitute a different encoder -- the forward()
                # only uses the learnable projector path, so it is unaffected.
                # predict_proba() callers must be tolerant of the missing
                # _tabpfn (they get a RuntimeError; F1.4 head does not call it).

    def training_step(self, train_batch, batch_idx):
        train_batch, corr, noise_ind = train_batch
        loss, accuracy, predictions, *extra = self._get_cross_entropy_and_accuracy(train_batch, noise_ind=noise_ind)
        self.log("Train/accuracy", accuracy, on_step=True, prog_bar=True)
        self.log("Train/loss", loss, on_step=True)
        self.log_metrics(predictions, train_batch[-1], "Train/", on_step=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        val_batch,  _, noise_ind = val_batch
        loss, accuracy, predictions, *extra = self._get_cross_entropy_and_accuracy(val_batch, noise_ind=noise_ind)
        self.log("Val/accuracy", accuracy, prog_bar=True)
        self.log("Val/loss", loss)
        self.log_metrics(predictions, val_batch[-1], "Val/")
        return loss
    
    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     batch,  _, noise_ind = batch
    #     loss, accuracy, predictions, unimodal_pred = self._get_cross_entropy_and_accuracy(batch, noise_ind=noise_ind)
    #     return unimodal_pred, batch[-1]
        
    

    def test_step(self, batch, batch_idx, dataloader_idx=2):
        batch,  _, noise_ind = batch
        loss, accuracy, predictions, unimodal_pred = self._get_cross_entropy_and_accuracy(batch, noise_ind=noise_ind)
        set_name = DATALOADER_ID_TO_SET_NAME[dataloader_idx]
        if(set_name == "test"):
            self.test_pred += [predictions]
            self.test_target += [batch[-1]]
            self.unimodal_test_pred += [unimodal_pred]
        self.log(f"Test/{set_name}_accuracy", accuracy, add_dataloader_idx=False)
        self.log_metrics(predictions, batch[-1], f"Test/{set_name}_", add_dataloader_idx=False)


                
    def log_metrics(self, probs, targets, mode='Train/', **kwargs):
        # Do not call self.metrics[name](probs, targets) and log the returned tensor:
        # for AUROC / F1 / Precision / Recall, that gives a per-batch value that
        # Lightning then averages with batch-size weights — which is mathematically
        # wrong for ranking-based metrics (was off by ~30x in the upstream repo;
        # see paper/outline/c2mf-reproduction-recipe.md §7.5 and thesis findings).
        # Instead: .update() the accumulator each batch, log the Metric INSTANCE
        # itself, and Lightning calls .compute() once at epoch end and .reset()
        # automatically.
        for metric_name in self.metrics:
            self.metrics[metric_name].update(probs.detach(), targets.detach())
            self.log(f"{mode}{metric_name}", self.metrics[metric_name], **kwargs)
        
    def _get_cross_entropy_and_accuracy(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    

    def on_test_end(self) -> None:
        self.test_pred   = torch.cat(self.test_pred).cpu().detach()
        self.test_target = torch.cat(self.test_target).cpu().detach()
        self.unimodal_test_pred = torch.cat(self.unimodal_test_pred, dim=1).cpu().detach()

        
        #create a folder to save the plots
        test_noise = 100
        calib_folder = 'calibration_plots'
        os.makedirs(f'{calib_folder}/{self.cfg.exp_setup}/{self.cfg.group_tag}/{self.cfg.experiment.name}/T{self.cfg.trial}/noise_severity_{self.cfg.noise_severity}/test_noise_{test_noise}', exist_ok=True)
        os.chdir(f'{calib_folder}/{self.cfg.exp_setup}/{self.cfg.group_tag}/{self.cfg.experiment.name}/T{self.cfg.trial}/noise_severity_{self.cfg.noise_severity}/test_noise_{test_noise}')

        print("\nGenerating Calibration Plot...")

        probabilities = self.test_pred

        

        
        # ------------------ Start of Single Calibration Plot Code ------------------
        print("\nGenerating Reliability Diagram...")
        # Find the maximum predicted probability for each sample
        max_probabilities, predicted_classes = torch.max(probabilities, dim=1)

        make_model_diagrams(self.test_pred, self.test_target, n_bins=10, prefix='multimodal')
        make_model_diagrams(self.unimodal_test_pred[0], self.test_target, n_bins=10, prefix = 'unimodal_0')
        make_model_diagrams(self.unimodal_test_pred[1], self.test_target, n_bins=10, prefix = 'unimodal_1')

        print("Confidence calibration plot saved as confidence_calibration_plot.png")
        print(f"Calibration plots saved as PNG files to calibration_plots/{self.cfg.experiment.name}/noise_severity_{self.cfg.noise_severity}")
        
        for modality, pred in enumerate(self.unimodal_test_pred):
            print(f"\nModality {modality}:")
            for metric_name in self.metrics:
                metric = self.metrics[metric_name].cpu()
                score = metric(pred, self.test_target)
                print("Test/", metric_name, score)
                
        
        print("\nmultimodal:")
        for metric_name in self.metrics:
            metric = self.metrics[metric_name].cpu()
            score = metric(self.test_pred, self.test_target)
            unimodal_1_score = metric(self.unimodal_test_pred[0], self.test_target)
            unimodal_2_score = metric(self.unimodal_test_pred[1], self.test_target)
            print("Test/", metric_name, score)

            # save_metric_json(metric_name,
            #                  group_tag = self.cfg.group_tag,
            #      experiment_name=self.cfg.experiment.name,
            #      noise_setting=self.cfg.noise_severity,
            #      run_id=self.cfg.trial,
            #      multimodal=score.item(),
            #      unimodal1=unimodal_1_score.item(),
            #      unimodal2=unimodal_2_score.item(),
            #      file= "scores_"+ self.cfg.exp_setup+".json")

        
    


class LateFusionClassifier(FusionModel):
    """
    Late fusion model. Outputs the probability distribution over a target variable given input from multiple modalities
    """

    def __init__(self, cfg: DictConfig, steps_per_epoch: int, name="LateFusionClassifier", ):
        super().__init__(cfg, name=name, steps_per_epoch=steps_per_epoch)

    def configure_metrics(self):
        # nn.ModuleDict registers the metrics as submodules so Lightning tracks
        # them: required for the log_metrics() pattern to call .compute() and
        # .reset() at epoch end correctly. Plain dict would lose Lightning's
        # auto-aggregation handling.
        self.metrics = nn.ModuleDict({
            'Accuracy':  torchmetrics.Accuracy(task="multiclass", average='micro', num_classes=self.cfg.experiment.dataset.num_classes),
            # 'Accuracy_macro':  torchmetrics.Accuracy(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
            'AUROC': torchmetrics.AUROC(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
            'Precision': torchmetrics.Precision(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
            'Recall': torchmetrics.Recall(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
            'F1Score': torchmetrics.F1Score(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
        })


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
        

        if(not self.cfg.joint_training or self.current_epoch <=5):
            predictions = predictions.detach()
            context = context.detach()
        predictions = self.head(predictions, context)

        
        # Criterion is NLL which takes logp( y | x)
        # NOTE: Don't use nn.CrossEntropyLoss because it expects unnormalized logits
        # and applies LogSoftmax first
        ll_y_g_x = predictions.log()
        
        loss += self.criterion(ll_y_g_x, labels)
        if(hasattr(self.head,"loss")):
            loss += self.head.loss[:,labels].mean()
        accuracy = (labels == ll_y_g_x.argmax(-1)).sum() / ll_y_g_x.shape[0]
        return loss, accuracy, predictions, unimodal_predictions

    
    

class LateFusionMultiLabelClassifier(FusionModel):
    """
    Late fusion model. Outputs the probability distribution over a target variable given input from multiple modalities
    """

    def __init__(self, cfg: DictConfig, steps_per_epoch: int, name="LateFusionMultiLabelClassifier", ):
        super().__init__(cfg, name=name, steps_per_epoch=steps_per_epoch)
        self.criterion = nn.BCELoss()
        self.accuracy = torchmetrics.Accuracy(task="multilabel", num_labels= self.cfg.experiment.dataset.num_classes)
        
    def configure_metrics(self):
        self.metrics = nn.ModuleDict({
            'AUROC': torchmetrics.AUROC(task="multilabel", average='weighted', num_labels= self.cfg.experiment.dataset.num_classes),
            'Precision': torchmetrics.Precision(task="multilabel", average='weighted', num_labels= self.cfg.experiment.dataset.num_classes),
            'Recall': torchmetrics.Recall(task="multilabel", average='weighted', num_labels= self.cfg.experiment.dataset.num_classes),
            'F1Score': torchmetrics.F1Score(task="multilabel", average='weighted', num_labels= self.cfg.experiment.dataset.num_classes),
            'Accuracy': torchmetrics.Accuracy(task="multilabel", average='weighted', num_labels= self.cfg.experiment.dataset.num_classes),
        })

    def _get_cross_entropy_and_accuracy(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        for unimodal_data, encoder, predictor in zip(data, self.encoders, self.predictors):
            embeddings += [encoder(unimodal_data)]
            unimodal_prediction = predictor(embeddings[-1])
            predictions += [unimodal_prediction.unsqueeze(1)]
            loss += self.criterion(unimodal_prediction, labels.to(unimodal_prediction.dtype))
        
        predictions = torch.cat(predictions, dim=1)
        if(not self.cfg.joint_training):
            predictions = predictions.detach()
        predictions = torch.cat([predictions.unsqueeze(-1),1-predictions.unsqueeze(-1)], dim=-1)
        
        if(self.cfg.experiment.head.threshold_input):
            predictions = predictions.argmax(dim=-1)
        
        predictions = self.head(predictions, embeddings)
        loss += self.criterion(predictions.to(torch.float64), labels.to(torch.float64))
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy, predictions
    


class EarlyFusionDiscriminative(FusionModel):
    """
    Early fusion model. Outputs the probability distribution over a target variable given input from multiple modalities
    """
    def __init__(self, cfg: DictConfig, steps_per_epoch: int, name="EarlyFusionDiscriminative" ):
        super().__init__(cfg, name=name, steps_per_epoch=steps_per_epoch)

    def configure_metrics(self):
        self.metrics = nn.ModuleDict({
            'AUROC': torchmetrics.AUROC(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
            'Precision': torchmetrics.Precision(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
            'Recall': torchmetrics.Recall(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
            'F1Score': torchmetrics.F1Score(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
        })

    def _get_cross_entropy_and_accuracy(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
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
        loss, embeddings = 0.0,  []
        for unimodal_data, encoder in zip(data, self.encoders):
            embeddings += [encoder(unimodal_data)] #[batch, 3, 512]
        embeddings = torch.cat(embeddings, dim=-1) #[batch, 3, 1024]]
        embeddings = embeddings.view(embeddings.shape[0], -1)  # Flatten the embeddings
        # print("Embeddings shape:", embeddings.shape)
        predictions = self.head(embeddings)
        # Criterion is NLL which takes logp( y | x)
        # NOTE: Don't use nn.CrossEntropyLoss because it expects unnormalized logits
        # and applies LogSoftmax first
        ll_y_g_x = predictions.log()
        loss += self.criterion(ll_y_g_x, labels)
        accuracy = (labels == ll_y_g_x.argmax(-1)).sum() / ll_y_g_x.shape[0]
        return loss, accuracy, predictions
