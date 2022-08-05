import pytorch_lightning as pl


NUMBER_OF_TAGS = 17


def test_amazone_datamodule(amazon_dm: pl.LightningDataModule):
    dm = amazon_dm
    dm.setup(stage='fit')
    imgs, tags = next(iter(dm.train_dataloader()))
    assert imgs.shape == (dm.batch_size, 3, dm.train_augs['img_size'], dm.train_augs['img_size'])
    assert tags.shape == (dm.batch_size, 1, NUMBER_OF_TAGS)
