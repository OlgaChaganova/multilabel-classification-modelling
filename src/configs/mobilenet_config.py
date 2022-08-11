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

NUM_CLASSES = 17
IMG_SIZE = 256

CONFIG = Config(
    project=Project(
        project_name='cvr-hw1-modelling',
        task_name='mobilenet_v3_small_ce',
    ),

    common=Common(seed=8),

    dataset=Dataset(
        root='/root/cvr-hw1-modeling/raw_data/train-jpg',
        imlist_filename='/root/cvr-hw1-modeling/raw_data/train_v2.csv',
        test_size=0.1,
        img_type='jpg',
        img_size=IMG_SIZE,
        num_channels=3,
        num_classes=NUM_CLASSES,
        batch_size=56,
        num_workers=6,
        train_augmentations='default',
        valid_augmentations='default',
    ),

    model=Model(
        model_params={
            'emb_size': 256,
            'backbone': 'mobilenet_v3_small',
            'dropout': 0.5,
            'num_classes': NUM_CLASSES,
            'num_channels': 3,
            'img_size': IMG_SIZE,
        },
    ),

    train=Train(
        trainer_params={
            'devices': 1,
            'accelerator': 'auto',
            'accumulate_grad_batches': 4,
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
                dirpath='/root/cvr-hw1-modeling/checkpoints/mobilenet_v3_small_ce/',
                save_top_k=3,
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
                'patience': 3,
                'factor': 0.1,
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
