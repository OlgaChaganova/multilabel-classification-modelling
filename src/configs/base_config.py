import typing as tp
from dataclasses import dataclass

import pytorch_lightning as pl
from torch import nn

from src.data.augmentations import AUGMENTATION_MODES
from src.data.dataset import IMG_TYPES


@dataclass
class Project:
    project_name: str
    task_name: str


@dataclass
class Common:
    seed: int = 8


@dataclass
class Dataset:
    root: str
    imlist_filename: str
    num_classes: int = 2000
    num_channels: int = 3
    img_type: IMG_TYPES = 'jpg'
    batch_size: int = 80
    img_size: int = 224
    test_size: float = 0.1
    train_augmentations: AUGMENTATION_MODES = 'default'
    valid_augmentations: AUGMENTATION_MODES = 'default'
    num_workers: int = 2


@dataclass
class Model:
    model_params: dict
    threshold: float


@dataclass
class Callbacks:
    model_checkpoint: pl.callbacks.ModelCheckpoint
    early_stopping: tp.Optional[pl.callbacks.EarlyStopping] = None
    lr_monitor: tp.Optional[pl.callbacks.LearningRateMonitor] = None
    model_summary: tp.Optional[tp.Union[pl.callbacks.ModelSummary, pl.callbacks.RichModelSummary]] = None
    timer: tp.Optional[pl.callbacks.Timer] = None


@dataclass
class Optimizer:
    name: str
    opt_params: dict


@dataclass
class LRScheduler:
    name: str
    lr_sched_params: dict


@dataclass
class Criterion:
    loss: nn.Module


@dataclass
class Train:
    trainer_params: dict
    callbacks: Callbacks
    optimizer: Optimizer
    lr_scheduler: LRScheduler
    criterion: Criterion
    ckpt_path: tp.Optional[str] = None


@dataclass
class Config:
    project: Project
    common: Common
    dataset: Dataset
    model: Model
    train: Train
