import typing as tp

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision import models

from src.configs.base_config import Optimizer, LRScheduler, Criterion

BACKBONES = tp.Literal[
    'densenet',
    'mobilenet_v3',
    'convnext',
    'efficientnet'
]


class MultiLabelClassifier(pl.LightningModule):
    def __init__(self,
                 emb_size: int,
                 backbone: BACKBONES,
                 dropout: float,
                 num_classes: int,
                 num_channels: int,
                 img_size: int,
                 optimizer: Optimizer,
                 lr_scheduler: LRScheduler,
                 criterion: Criterion):
        """
        Base feature extractor model that can be combined with different loss functions

        :param emb_size: embeddings of input images size
        :param backbone: backbone architecture
        :param dropout: dropout rate in Dropout layers
        :param num_classes: number of classes in dataset
        :param num_channels: number of input images channels
        :param img_size: size of input images
        :param optimizer: optimizer (name and parameters)
        :param lr_scheduler: learning rate scheduler (name and parameters)
        :param criterion: loss function for classification
        """

        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = optimizer.params['lr']
        self.emb_size = emb_size

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.criterion = criterion

        self._build_backbone(backbone)
        self._set_size(num_channels, img_size)

        self.normalize_conv = nn.Sequential(
            nn.BatchNorm2d(self.backbone_in_features),
            nn.Dropout2d(dropout, inplace=True)
        )

        self.normalize_lin = nn.Sequential(
            nn.Linear(in_features=(self.backbone_in_features * self.features_shape[-2] * self.features_shape[-1]),
                      out_features=self.emb_size),
            nn.BatchNorm1d(self.emb_size)
        )

        self.classifier = nn.Linear(in_features=self.emb_size,
                                    out_features=num_classes)

        self.sigmoid = nn.Sigmoid()

    def _build_backbone(self, arch: str):
        if arch.startswith('densenet'):
            weights = arch.capitalize().replace('net', 'Net')
            weights = f'{weights}_Weights'
            self.backbone = models.__dict__[arch](weights=models.__dict__[weights].IMAGENET1K_V1)
            self.backbone_in_features = self.backbone.classifier.in_features

        elif arch.startswith('mobilenet_v3'):
            weights = arch.capitalize().replace('net', 'Net')
            weights = weights.replace('_v', '_V').replace('_s', '_S').replace('_l', '_L')
            weights = f'{weights}_Weights'
            self.backbone = models.__dict__[arch](weights=models.__dict__[weights].IMAGENET1K_V1)
            self.backbone_in_features = self.backbone.classifier[0].in_features

        elif arch.startswith('convnext'):
            weights = arch.capitalize().replace('next', 'NeXt')
            weights = weights.replace('_b', '_B').replace('_s', '_S').replace('_l', '_L').replace('_t', '_T')
            weights = f'{weights}_Weights'
            self.backbone = models.__dict__[arch](weights=models.__dict__[weights].IMAGENET1K_V1)
            self.backbone_in_features = self.backbone.classifier[-1].in_features

        elif arch.startswith('efficientnet'):
            weights = arch.capitalize().replace('net', 'Net').replace('_b', '_B')
            weights = f'{weights}_Weights'
            self.backbone = models.__dict__[arch](weights=models.__dict__[weights].IMAGENET1K_V1)
            self.backbone_in_features = self.backbone.classifier[-1].in_features

        else:
            raise ValueError(
                f'Available backbone types are '
                f'{", ".join(tp.get_args(BACKBONES))}, but got {arch}'
            )

    def _set_size(self, num_channels: int, img_size: int):
        dummy_inp = torch.rand([1, num_channels, img_size, img_size])
        self.features_shape = self.backbone.features(dummy_inp).shape

    def forward(self, x):
        features = self.backbone.features(x)
        features = self.normalize_conv(features)
        features = features.contiguous() .view(features.size(0), -1)
        embeddings = self.normalize_lin(features)
        logits = self.classifier(embeddings)
        probs = self.sigmoid(logits)
        return probs

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1)

        probs = self.forward(x)
        loss = self.criterion(probs, y)

        # acc = calc_accuracy(logits, y)

        self.log('train_loss', loss, on_epoch=True, on_step=True)
        # self.log('train_acc', acc, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1)

        probs = self.forward(x)
        loss = self.criterion(probs, y)

        # acc = calc_accuracy(logits, y)
        self.log('val_loss', loss, on_epoch=True, on_step=True)
        # self.log('val_acc', acc, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer.name)(self.parameters(), **self.optimizer.params)
        lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler.name)(optimizer, **self.lr_scheduler.params)
        return {'optimizer': optimizer,
                'lr_scheduler': lr_scheduler,
                'monitor': 'val_loss'}
