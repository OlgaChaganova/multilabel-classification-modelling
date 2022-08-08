import pytorch_lightning as pl
import torch
from clearml import Task
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from src.configs.base_config import Config


NUMBER_OF_TAGS = 17


def test_build_model(model: pl.LightningModule):
    model


def test_model_forward_pass(model: pl.LightningModule):
    batch_size, num_channels, img_size = 10, 3, 224
    x = torch.randn([batch_size, num_channels, img_size, img_size])
    output = model(x)
    assert output.shape == (batch_size, NUMBER_OF_TAGS)


def test_model_train_step(model: pl.LightningModule):
    batch_size, num_channels, img_size = 10, 3, 224
    x = torch.randn([batch_size, num_channels, img_size, img_size])
    y = torch.randn([batch_size, NUMBER_OF_TAGS])
    probs = model(x)
    loss = model.criterion(probs, y)
    loss.backward()


def test_train_loop(
        config: Config,
        model: pl.LightningModule,
        amazon_dm: pl.LightningDataModule
):
    # task in ClearML
    task = Task.init(project_name=config.info.project_name, task_name=config.info.task_name)

    # trainer
    callbacks = list(config.train.callbacks.__dict__.values())
    callbacks = filter(lambda x: x is not None, callbacks)
    trainer = Trainer(
        callbacks=[
            TQDMProgressBar(refresh_rate=1),
            RichModelSummary(),
            *callbacks
        ],
        fast_dev_run=7,
        log_every_n_steps=2,
        **config.train.trainer_params
    )

    if config.train.trainer_params['auto_scale_batch_size'] is not None or \
            config.train.trainer_params['auto_lr_find'] is not None:
        trainer.tune(model=model, datamodule=amazon_dm)

    trainer.fit(
        model=model,
        datamodule=amazon_dm,
        ckpt_path=config.train.ckpt_path,
    )
