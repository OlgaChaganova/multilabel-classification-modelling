import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from src.configs.base_config import Config


def test_train_loop(
    config: Config,
    model: pl.LightningModule,
    amazon_dm: pl.LightningDataModule,
):
    trainer_params = config.train.trainer_params
    callbacks = list(config.train.callbacks.__dict__.values())
    callbacks = filter(lambda callback: callback is not None, callbacks)

    trainer = Trainer(
        callbacks=[
            TQDMProgressBar(refresh_rate=1),
            RichModelSummary(),
            *callbacks,
        ],
        fast_dev_run=5,
        log_every_n_steps=1,
        **trainer_params,
    )

    trainer.fit(
        model=model,
        datamodule=amazon_dm,
        ckpt_path=config.train.ckpt_path,
    )
