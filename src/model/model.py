import typing as tp

import pytorch_lightning as pl
import torch
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall
from torchvision import models

from src.configs.base_config import LRScheduler, Optimizer

BACKBONES = tp.Literal[
    'densenet',
    'mobilenet_v3',
    'convnext',
    'efficientnet',
]


class MultiLabelClassifier(pl.LightningModule):
    def __init__(
        self,
        emb_size: int,
        backbone: BACKBONES,
        dropout: float,
        num_classes: int,
        num_channels: int,
        img_size: int,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None,
        criterion: torch.nn.Module,
    ):
        """Create base feature extractor model that can be combined with different loss functions.

        Parameters
        ----------
        emb_size : int
            Embeddings of input images size.
        backbone : BACKBONES
            Backbone architecture.
        dropout : float
            Dropout rate in Dropout layers.
        num_classes : int
            Number of classes in dataset.
        num_channels : int
            Number of input images channels.
        img_size : int
            Size of input images.
        optimizer : Optimizer
            Optimizer.
        lr_scheduler : LRScheduler
            Learning rate schediler.
        criterion : torch.nnModule
            Loss function for classification.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['criterion'])  # criterion is already saved during checkpointing
        self.learning_rate = optimizer.opt_params['lr']
        self.emb_size = emb_size
        self.num_classes = num_classes

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.criterion = criterion

        self._build_backbone(backbone)
        self._set_size(num_channels, img_size)

        self.normalize_conv = torch.nn.Sequential(
            torch.nn.BatchNorm2d(self.backbone_in_features),
            torch.nn.Dropout2d(dropout, inplace=True),
        )

        self.normalize_lin = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=(self.backbone_in_features * self.features_shape[-2] * self.features_shape[-1]),
                out_features=self.emb_size,
            ),
            torch.nn.BatchNorm1d(self.emb_size),
        )

        self.classifier = torch.nn.Linear(
            in_features=self.emb_size,
            out_features=num_classes,
        )

        self.sigmoid = torch.nn.Sigmoid()

        # metrics
        self.accuracy = Accuracy()
        self.auroc = AUROC(num_classes=num_classes)
        self.f1_score = F1Score(num_classes=num_classes)

    def forward(self, imgs: torch.tensor):
        features = self.backbone.features(imgs)
        features = self.normalize_conv(features)
        features = features.contiguous() .view(features.size(0), -1)
        embeddings = self.normalize_lin(features)
        logits = self.classifier(embeddings)
        probs = self.sigmoid(logits)
        return probs  # noqa: WPS331

    def training_step(self, batch, batch_idx):
        imgs, tags = batch
        tags = tags.squeeze(1)
        probs = self.forward(imgs)

        loss = self.criterion(probs, tags)
        self.log('train_loss', loss, on_epoch=True, on_step=True)

        self.accuracy(probs, tags.long())
        self.log('train_acc', self.accuracy, on_epoch=True, on_step=True)

        self.auroc(probs, tags.long())
        self.log('train_auroc', self.auroc, on_epoch=True, on_step=True)

        self.f1_score(probs, tags.long())
        self.log('train_f1', self.f1_score, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, tags = batch
        tags = tags.squeeze(1)
        probs = self.forward(imgs)

        loss = self.criterion(probs, tags)
        self.log('val_loss', loss, on_epoch=True, on_step=True)

        self.accuracy(probs, tags.long())
        self.log('val_acc', self.accuracy, on_epoch=True, on_step=True)

        self.auroc(probs, tags.long())
        self.log('val_auroc', self.auroc, on_epoch=True, on_step=True)

        self.f1_score(probs, tags.long())
        self.log('val_f1', self.f1_score, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer.name)(
            self.parameters(),
            **self.optimizer.opt_params,
        )

        optim_dict = {
            'optimizer': optimizer,
            'monitor': 'val_loss',
        }

        if self.lr_scheduler is not None:
            lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler.name)(
                optimizer,
                **self.lr_scheduler.lr_sched_params,
            )

            optim_dict.update({'lr_scheduler': lr_scheduler})

        return optim_dict

    def _build_backbone(self, arch: str):
        if arch.startswith(tp.get_args(BACKBONES)):
            backbone = getattr(models, arch)(weights=None)

            if arch.startswith('densenet'):
                weights = arch.capitalize().replace('net', 'Net')
                self.backbone_in_features = backbone.classifier.in_features

            elif arch.startswith('mobilenet_v3'):
                weights = arch.capitalize().replace('net', 'Net')
                weights = weights.replace('_v', '_V')
                weights = weights.replace('_s', '_S')
                weights = weights.replace('_l', '_L')
                self.backbone_in_features = backbone.classifier[0].in_features

            elif arch.startswith('convnext'):
                weights = arch.capitalize().replace('next', 'NeXt')
                weights = weights.replace('_b', '_B')
                weights = weights.replace('_s', '_S')
                weights = weights.replace('_l', '_L')
                weights = weights.replace('_t', '_T')
                self.backbone_in_features = backbone.classifier[-1].in_features

            elif arch.startswith('efficientnet'):
                weights = arch.capitalize().replace('net', 'Net')
                weights = weights.replace('_b', '_B')
                self.backbone_in_features = backbone.classifier[-1].in_features

            weights = f'{weights}_Weights'
            self.backbone = getattr(models, arch)(weights=getattr(models, weights).IMAGENET1K_V1)

        else:
            available_backbones = tp.get_args(BACKBONES)
            raise ValueError(
                f'Available backbone types are {available_backbones}, but got {arch}',
            )

    def _set_size(self, num_channels: int, img_size: int):
        dummy_inp = torch.rand([1, num_channels, img_size, img_size])
        self.features_shape = self.backbone.features(dummy_inp).shape
