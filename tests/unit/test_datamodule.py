import os
import pytorch_lightning as pl

from src.configs.base_config import Config


NUM_CLASSES = 17


def test_datamodule(amazon_dm: pl.LightningDataModule, config: Config):
    amazon_dm.setup(stage='fit')
    dataloader = amazon_dm.train_dataloader()
    img, tags = next(iter(dataloader))

    batch_size = config.dataset.batch_size
    num_channels = config.dataset.num_channels
    img_size = config.dataset.img_size
    assert img.shape == (batch_size, num_channels, img_size, img_size)
    assert tags.shape == (batch_size, 1, NUM_CLASSES)
