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
from omegaconf import open_dict
from models.losses import FLandMDCA
from models.utils import save_metric_json, make_model_diagrams
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
        self.head = eval(cfg.experiment.head.type)(**cfg.experiment.head.args)
        self.encoders, self.predictors = torch.nn.ModuleList(self.encoders), torch.nn.ModuleList(self.predictors)
        self.num_modalities = cfg.experiment.dataset.modalities
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

    def training_step(self, train_batch, batch_idx):
        train_batch, _, noise_ind = train_batch
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
        probs, targets = probs.detach().cpu(), targets.detach().cpu()
        for metric_name in self.metrics:
            self.log(f"{mode}{metric_name}",self.metrics[metric_name](probs, targets),**kwargs)
        
    def _get_cross_entropy_and_accuracy(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    

    def on_test_end(self) -> None:
        self.test_pred   = torch.cat(self.test_pred).cpu().detach()
        self.test_target = torch.cat(self.test_target).cpu().detach()
        self.unimodal_test_pred = torch.cat(self.unimodal_test_pred, dim=1).cpu().detach()

        
        #create a folder to save the plots
        calib_folder = 'calibration_plots'
        os.makedirs(f'{calib_folder}/{self.cfg.exp_setup}/{self.cfg.group_tag}/{self.cfg.experiment.name}/T{self.cfg.trial}/noise_severity_{self.cfg.noise_severity}', exist_ok=True)
        os.chdir(f'{calib_folder}/{self.cfg.exp_setup}/{self.cfg.group_tag}/{self.cfg.experiment.name}/T{self.cfg.trial}/noise_severity_{self.cfg.noise_severity}')

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


        # print("Class wise accuracy:")
        # for modality, pred in enumerate(self.unimodal_test_pred):
        #     print(f"\nModality {modality}:")
        #     acc_class_wise = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.cfg.experiment.dataset.num_classes, average='none')
        #     acc_class_wise = acc_class_wise(pred, self.test_target)
        #     for i in range(self.cfg.experiment.dataset.num_classes):
        #         print(f"Class {i} accuracy: {acc_class_wise[i].item():.4f}")
        #     print(f"Overall accuracy:")
        #     print(f"Micro: {torchmetrics.classification.Accuracy(task='multiclass', num_classes=self.cfg.experiment.dataset.num_classes, average= 'micro')(pred, self.test_target).item():.4f}")
        #     print(f"Macro: {torchmetrics.classification.Accuracy(task='multiclass', num_classes=self.cfg.experiment.dataset.num_classes, average= 'macro')(pred, self.test_target).item():.4f}")
        # print("\nMultimodal:")
        # acc_class_wise = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.cfg.experiment.dataset.num_classes, average='none')
        # acc_class_wise = acc_class_wise(self.test_pred, self.test_target)
        # for i in range(self.cfg.experiment.dataset.num_classes):
        #     print(f"Class {i} accuracy: {acc_class_wise[i].item():.4f}")
        
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

            save_metric_json(metric_name,
                             group_tag = self.cfg.group_tag,
                 experiment_name=self.cfg.experiment.name,
                 noise_setting=self.cfg.noise_severity,
                 run_id=self.cfg.trial,
                 multimodal=score.item(),
                 unimodal1=unimodal_1_score.item(),
                 unimodal2=unimodal_2_score.item(),
                 file= "scores_"+ self.cfg.exp_setup+".json")

        
    


class LateFusionClassifier(FusionModel):
    """
    Late fusion model. Outputs the probability distribution over a target variable given input from multiple modalities
    """

    def __init__(self, cfg: DictConfig, steps_per_epoch: int, name="LateFusionClassifier", ):
        super().__init__(cfg, name=name, steps_per_epoch=steps_per_epoch)

    def configure_metrics(self):
        self.metrics = {
            'Accuracy':  torchmetrics.Accuracy(task="multiclass", average='micro', num_classes=self.cfg.experiment.dataset.num_classes),
            # 'Accuracy_macro':  torchmetrics.Accuracy(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
            'AUROC': torchmetrics.AUROC(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
            'Precision': torchmetrics.Precision(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
            'Recall': torchmetrics.Recall(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
            'F1Score': torchmetrics.F1Score(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
        }

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
            loss += self.criterion(ll_y_g_x, labels)

        predictions = torch.cat(predictions, dim=1)
        unimodal_predictions = torch.permute(predictions, (1,0,2))
        
        if (hasattr(self.cfg.experiment.head, "noise_context") and not self.cfg.experiment.head.noise_context):
            context = torch.cat(embeddings, dim=-1)

        else:
            context = [torch.cat((t1, t2), dim=-1) for t1, t2 in zip(embeddings, corruptions)]
            context = torch.cat(context, dim=-1)



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
        self.metrics = {
            'AUROC': torchmetrics.AUROC(task="multilabel", average='weighted', num_labels= self.cfg.experiment.dataset.num_classes),
            'Precision': torchmetrics.Precision(task="multilabel", average='weighted', num_labels= self.cfg.experiment.dataset.num_classes),
            'Recall': torchmetrics.Recall(task="multilabel", average='weighted', num_labels= self.cfg.experiment.dataset.num_classes),
            'F1Score': torchmetrics.F1Score(task="multilabel", average='weighted', num_labels= self.cfg.experiment.dataset.num_classes),
            'Accuracy': torchmetrics.Accuracy(task="multilabel", average='weighted', num_labels= self.cfg.experiment.dataset.num_classes),
        }

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
        self.metrics = {
            'AUROC': torchmetrics.AUROC(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
            'Precision': torchmetrics.Precision(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
            'Recall': torchmetrics.Recall(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
            'F1Score': torchmetrics.F1Score(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
        }

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