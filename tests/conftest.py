import pytest
import pytorch_lightning as pl
from torch import nn

from src.configs.base_config import (
    Callbacks,
    Common,
    Config,
    Criterion,
    Dataset,
    LRScheduler,
    Model,
    Optimizer,
    Project,
    Train,
)
from src.data.dataset import AmazonDataModule
from src.model.model import MultiLabelClassifier


NUM_CLASSES = 17
IMG_SIZE = 256


@pytest.fixture
def config():
    config = Config(
        project=Project(
            project_name='cvr-hw1-modelling',
            task_name='densenet121_ce',
        ),

        common=Common(seed=8),

        dataset=Dataset(
            root='tests/supplementary/amazon_space_dataset/images/',
            imlist_filename='tests/supplementary/amazon_space_dataset/train_v2.csv',
            test_size=0.1,
            img_type='jpg',
            img_size=IMG_SIZE,
            num_channels=3,
            num_classes=NUM_CLASSES,
            batch_size=2,
            num_workers=2,
            train_augmentations='default',
            valid_augmentations='default',
            path_label_encoder_classes='./weights/label_encoder_classes.npy',
        ),

        model=Model(
            model_params={
                'emb_size': 512,
                'backbone': 'densenet121',
                'dropout': 0.5,
                'num_classes': NUM_CLASSES,
                'num_channels': 3,
                'img_size': IMG_SIZE,
            },
            threshold=0.5,
        ),

        train=Train(
            trainer_params={
                'devices': 1,
                'accelerator': 'auto',
                'accumulate_grad_batches': 1,
                'auto_scale_batch_size': None,
                'gradient_clip_val': 0,
                'benchmark': True,
                'precision': 32,
                'profiler': 'simple',
                'max_epochs': 10,
                'auto_lr_find': None,
            },

            callbacks=Callbacks(
                model_checkpoint=pl.callbacks.ModelCheckpoint(
                    dirpath=None,
                    save_top_k=1,
                    monitor='val_loss',
                    mode='min',
                ),

                lr_monitor=pl.callbacks.LearningRateMonitor(logging_interval='step'),
            ),

            optimizer=Optimizer(
                name='Adam',
                opt_params={
                    'lr': 0.001,
                    'weight_decay': 0.0001,
                },
            ),

            lr_scheduler=LRScheduler(
                name='ReduceLROnPlateau',
                lr_sched_params={
                    'patience': 5,
                    'factor': 0.5,
                    'mode': 'min',
                    'min_lr': 0.00001,
                },
            ),

            criterion=Criterion(
                loss=nn.BCELoss(),
            ),
            ckpt_path=None,
        ),
    )
    return config


@pytest.fixture
def amazon_dm(config):
    dm = AmazonDataModule(
        imlist_filename=config.dataset.imlist_filename,
        root=config.dataset.root,
        batch_size=config.dataset.batch_size,
        img_type=config.dataset.img_type,
        img_size=config.dataset.img_size,
        test_size=config.dataset.test_size,
        train_aug_mode=config.dataset.train_augmentations,
        valid_aug_mode=config.dataset.valid_augmentations,
        num_workers=config.dataset.num_workers,
    )
    return dm


@pytest.fixture
def model(config):
    model = MultiLabelClassifier(
        optimizer=config.train.optimizer,
        lr_scheduler=config.train.lr_scheduler,
        criterion=config.train.criterion.loss,
        **config.model.model_params,
    )
    return model
