import pytorch_lightning as pl
import torch
from clearml import Task
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from src.configs.base_config import Config, LRScheduler, Optimizer
from src.model.model import MultiLabelClassifier

NUMBER_OF_TAGS = 17


def test_model_densenet():
    batch_size, num_channels, img_size = 10, 3, 256
    model = MultiLabelClassifier(
        emb_size=512,
        backbone='densenet121',
        dropout=0.5,
        num_classes=NUMBER_OF_TAGS,
        num_channels=num_channels,
        img_size=img_size,
        optimizer=Optimizer(
            name='Adam',
            opt_params={
                'lr': 0.001,
            },
        ),
        lr_scheduler=LRScheduler(
            name='StepLR',
            lr_sched_params={
                'step_size': 10,
            },
        ),
        criterion=torch.nn.CrossEntropyLoss(),
    )
    imgs = torch.randn([batch_size, num_channels, img_size, img_size])
    output = model(imgs)
    assert output.shape == (batch_size, NUMBER_OF_TAGS)


def test_model_mobilenet_v3():
    batch_size, num_channels, img_size = 10, 3, 256
    model = MultiLabelClassifier(
        emb_size=512,
        backbone='mobilenet_v3_small',
        dropout=0.5,
        num_classes=NUMBER_OF_TAGS,
        num_channels=num_channels,
        img_size=img_size,
        optimizer=Optimizer(
            name='Adam',
            opt_params={
                'lr': 0.001,
            },
        ),
        lr_scheduler=LRScheduler(
            name='StepLR',
            lr_sched_params={
                'step_size': 10,
            },
        ),
        criterion=torch.nn.CrossEntropyLoss(),
    )
    imgs = torch.randn([batch_size, num_channels, img_size, img_size])
    output = model(imgs)
    assert output.shape == (batch_size, NUMBER_OF_TAGS)


def test_build_model_convnext():
    batch_size, num_channels, img_size = 10, 3, 256
    model = MultiLabelClassifier(
        emb_size=512,
        backbone='convnext_tiny',
        dropout=0.5,
        num_classes=NUMBER_OF_TAGS,
        num_channels=num_channels,
        img_size=img_size,
        optimizer=Optimizer(
            name='Adam',
            opt_params={
                'lr': 0.001,
            },
        ),
        lr_scheduler=LRScheduler(
            name='StepLR',
            lr_sched_params={
                'step_size': 10,
            },
        ),
        criterion=torch.nn.CrossEntropyLoss(),
    )
    imgs = torch.randn([batch_size, num_channels, img_size, img_size])
    output = model(imgs)
    assert output.shape == (batch_size, NUMBER_OF_TAGS)


def test_build_model_efficientnet():
    batch_size, num_channels, img_size = 10, 3, 256
    model = MultiLabelClassifier(
        emb_size=768,
        backbone='efficientnet_b4',
        dropout=0.3,
        num_classes=NUMBER_OF_TAGS,
        num_channels=num_channels,
        img_size=img_size,
        optimizer=Optimizer(
            name='Adam',
            opt_params={
                'lr': 0.001,
            },
        ),
        lr_scheduler=LRScheduler(
            name='StepLR',
            lr_sched_params={
                'step_size': 10,
            },
        ),
        criterion=torch.nn.CrossEntropyLoss(),
    )
    imgs = torch.randn([batch_size, num_channels, img_size, img_size])
    output = model(imgs)
    assert output.shape == (batch_size, NUMBER_OF_TAGS)


def test_model_train_step(model: pl.LightningModule):
    batch_size, num_channels, img_size = 10, 3, 224
    imgs = torch.randn([batch_size, num_channels, img_size, img_size])
    tags = torch.randn([batch_size, NUMBER_OF_TAGS])
    probs = model(imgs)
    loss = model.criterion(probs, tags)
    loss.backward()


def test_train_loop(
    config: Config,
    model: pl.LightningModule,
    amazon_dm: pl.LightningDataModule,
):
    # task in ClearML
    Task.init(project_name=config.project.project_name, task_name=config.project.task_name)

    # trainer
    trainer_params = config.train.trainer_params
    callbacks = list(config.train.callbacks.__dict__.values())
    callbacks = filter(lambda callback: callback is not None, callbacks)
    trainer = Trainer(
        callbacks=[
            TQDMProgressBar(refresh_rate=1),
            RichModelSummary(),
            *callbacks,
        ],
        fast_dev_run=7,
        log_every_n_steps=2,
        **config.train.trainer_params,
    )

    if trainer_params['auto_scale_batch_size'] is not None or trainer_params['auto_lr_find'] is not None:
        trainer.tune(model=model, datamodule=amazon_dm)

    trainer.fit(
        model=model,
        datamodule=amazon_dm,
        ckpt_path=config.train.ckpt_path,
    )
