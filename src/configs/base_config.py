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
    root: str  # path to root directory with images
    imlist_filename: str  # path to image list with paths to images and their labels
    num_classes: int  # number of classes in dataset
    num_channels: int  # number of channels of images _after_ transforming
    img_type: IMG_TYPES  # image file extension
    batch_size: int
    img_size: int  # size of images _after_ transforming
    test_size: float  # share of validation dataset
    train_augmentations: AUGMENTATION_MODES
    valid_augmentations: AUGMENTATION_MODES
    num_workers: int
    path_label_encoder_classes: str  # path to file where label encoder classes will be stored


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
