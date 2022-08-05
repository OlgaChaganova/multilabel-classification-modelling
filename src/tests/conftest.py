import pytest

from src.data.dataset import AmazonDataModule


@pytest.fixture
def amazon_dm():
    dm = AmazonDataModule(
        imlist_filename='/home/olga/PycharmProjects/cvr-hw1-modeling/raw_data/train_v2.csv',
        root='/home/olga/PycharmProjects/cvr-hw1-modeling/raw_data/train-jpg',
        batch_size=16,
        img_type='jpg',
        img_size=256,
        test_size=0.1,
        train_aug_mode='default',
        valid_aug_mode='default',
        num_workers=6
    )
    return dm

