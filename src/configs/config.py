import pytorch_lightning as pl
from base_config import Callbacks, Common, Config, Criterion, Dataset, LRScheduler, Model, Optimizer, Project, Train
from torch import nn

CONFIG = Config(
    project=Project(
        project_name='cvr-hw1-modelling',
        task_name='densenet121_ce',
    ),

    common=Common(seed=8),

    dataset=Dataset(
        root='/home/olga/PycharmProjects/cvr-hw1-modeling/raw_data/train-jpg',
        imlist_filename='/home/olga/PycharmProjects/cvr-hw1-modeling/raw_data/train_v2.csv',
        test_size=0.1,
        img_type='jpg',
        img_size=224,
        num_channels=3,
        num_classes=2000,
        batch_size=64,
        num_workers=2,
        train_augmentations='default',
        valid_augmentations='default',
    ),

    model=Model(
        model_params={
            'emb_size': 512,
            'backbone': 'densenet121',
            'dropout': 0.5,
            'num_classes': 2000,
            'num_channels': 3,
            'img_size': 224,
        },
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
                dirpath='/content/face-research/src/models/feature_extractor/experiments/',
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
