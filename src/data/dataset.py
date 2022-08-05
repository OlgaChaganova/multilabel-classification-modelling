import os
import typing as tp

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader

from .augmentations import AUGMENTATION_MODES
from .augmentations import get_val_aug, get_train_aug


IMG_TYPES = tp.Literal[
    'jpg',
    'jpeg',
    'png'
]


def get_imlist(imlist_filename: str, sep: str = ',') -> pd.DataFrame:
    imlist = pd.read_table(imlist_filename, sep=sep)
    imlist['tags'] = imlist['tags'].apply(lambda x: x.split()).tolist()
    return imlist


def get_label_encoder(labels_list: tp.List[tp.List[str]]) -> MultiLabelBinarizer:
    label_encoder = MultiLabelBinarizer()
    label_encoder.fit(labels_list)
    return label_encoder


class AmazonDataset(Dataset):
    def __init__(self,
                 root: str,
                 img_type: str,
                 imlist: pd.DataFrame,
                 label_encoder: MultiLabelBinarizer,
                 transforms: T.Compose):

        self.root = root
        self.img_type = img_type
        self.imlist = imlist
        self.label_encoder = label_encoder
        self.transforms = transforms

    def __len__(self):
        return len(self.imlist)

    def load_sample(self, filename):
        filename += ('.' + self.img_type)
        filename = os.path.join(self.root, filename)
        image = Image.open(filename)
        image = image.convert("RGB")
        image.load()
        return image

    def __getitem__(self, ind):
        filename, tags = self.imlist.iloc[ind, :]
        x = self.load_sample(filename)
        x = self.transforms(x)
        y = self.label_encoder.transform([tags]).astype(np.int64)
        return x, y


class AmazonDataModule(pl.LightningDataModule):
    def __init__(self,
                 imlist_filename: str,
                 root: str,
                 batch_size: int,
                 img_type: IMG_TYPES,
                 img_size: int,
                 test_size: float,
                 train_aug_mode: AUGMENTATION_MODES,
                 valid_aug_mode: AUGMENTATION_MODES,
                 num_workers: int):
        """
        Data Module for Amazon Competition.

        :param imlist_filename: name of csv file with paths to images and labels
        :param root: path to root dir with dataset images
        :param batch_size: batch size
        :param img_type: image file extension (jpg, jpeg, png)
        :param img_size: size of images after transforming and resizing
        :param test_size: size of val / test
        :param train_aug_mode: train augmentations mode
        :param valid_aug_mode: validation augmentation mode
        :param num_workers: number of workers in dataloaders
        """
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size

        self.root = root
        self.img_type = img_type
        self.train_augs = {'mode': train_aug_mode,
                           'img_size': img_size}
        self.val_augs = {'mode': valid_aug_mode,
                         'img_size': img_size}

        imlist = get_imlist(imlist_filename)
        self.train_imlist, self.val_imlist = train_test_split(imlist, test_size=test_size)
        self.label_encoder = get_label_encoder(imlist['tags'])

        self.num_workers = num_workers

    def setup(self, stage: tp.Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = AmazonDataset(
                root=self.root,
                img_type=self.img_type,
                imlist=self.train_imlist,
                transforms=get_train_aug(**self.train_augs),
                label_encoder=self.label_encoder
            )

            print(f'Mode: train, number of files: {self.train_dataset.__len__()}')

            self.val_dataset = AmazonDataset(
                root=self.root,
                img_type=self.img_type,
                imlist=self.val_imlist,
                transforms=get_val_aug(**self.train_augs),
                label_encoder=self.label_encoder
            )

            print(f'Mode: val, number of files: {self.val_dataset.__len__()}')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False
        )
