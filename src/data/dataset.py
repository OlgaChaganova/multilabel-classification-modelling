import logging
import os
import typing as tp

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tt

from src.data.augmentations import AUGMENTATION_MODES, get_train_aug, get_val_aug

IMG_TYPES = tp.Literal[
    'jpg',
    'jpeg',
    'png',
]


def get_imlist(imlist_filename: str, sep: str = ',', tag_column: str = 'tags') -> pd.DataFrame:
    imlist = pd.read_table(imlist_filename, sep=sep)
    imlist[tag_column] = imlist[tag_column].apply(lambda row: row.split())
    imlist[tag_column] = imlist[tag_column].tolist()
    return imlist


def get_label_encoder(labels_list: tp.List[tp.List[str]]) -> MultiLabelBinarizer:
    label_encoder = MultiLabelBinarizer()
    label_encoder.fit(labels_list)
    return label_encoder


class AmazonDataset(Dataset):
    def __init__(
        self,
        root: str,
        img_type: str,
        imlist: pd.DataFrame,
        label_encoder: MultiLabelBinarizer,
        transforms: tt.Compose,
    ):

        self.root = root
        self.img_type = img_type
        self.imlist = imlist
        self.label_encoder = label_encoder
        self.transforms = transforms

    def __len__(self):
        return len(self.imlist)

    def load_sample(self, filename):
        img_ext = '.{0}'.format(self.img_type)
        filename += img_ext
        filename = os.path.join(self.root, filename)
        image = Image.open(filename)
        image = image.convert('RGB')
        image.load()
        return image

    def __getitem__(self, ind):
        filename, tags = self.imlist.iloc[ind, :]
        img = self.load_sample(filename)
        img = self.transforms(img)
        tags = self.label_encoder.transform([tags]).astype(np.float32)
        return img, tags


class AmazonDataModule(pl.LightningDataModule):
    def __init__(
        self,
        imlist_filename: str,
        root: str,
        batch_size: int,
        img_type: IMG_TYPES,
        img_size: int,
        test_size: float,
        train_aug_mode: AUGMENTATION_MODES,
        valid_aug_mode: AUGMENTATION_MODES,
        num_workers: int,
    ):
        """Create Data Module for Amazon Competition.

        Parameters
        ----------
        imlist_filename : str
            Name of csv file with paths to images and labels.
        root : str
            Path to root dir with dataset images.
        batch_size : int
            Batch size for dataloaders.
        img_type : IMG_TYPES
            Image file extension (jpg, jpeg, png).
        img_size : int
            Size of images after transforming and resizing.
        test_size : float
            Size of valid and test datasets.
        train_aug_mode : AUGMENTATION_MODES
            Train augmentations mode.
        valid_aug_mode : AUGMENTATION_MODES
            Valid augmentations mode.
        num_workers : int
            Number of workers in dataloaders.

        """
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size

        self.root = root
        self.img_type = img_type
        self.train_augs = {
            'mode': train_aug_mode,
            'img_size': img_size,
        }
        self.val_augs = {
            'mode': valid_aug_mode,
            'img_size': img_size,
        }
        self.num_workers = num_workers

        imlist = get_imlist(imlist_filename)
        train_imlist, val_imlist = train_test_split(imlist, test_size=test_size)
        self.train_imlist = train_imlist
        self.val_imlist = val_imlist

        self.label_encoder = get_label_encoder(imlist['tags'])

    def setup(self, stage: tp.Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = AmazonDataset(
                root=self.root,
                img_type=self.img_type,
                imlist=self.train_imlist,
                transforms=get_train_aug(**self.train_augs),
                label_encoder=self.label_encoder,
            )
            num_train_files = len(self.train_dataset)
            logging.info(f'Mode: train, number of files: {num_train_files}')

            self.val_dataset = AmazonDataset(
                root=self.root,
                img_type=self.img_type,
                imlist=self.val_imlist,
                transforms=get_val_aug(**self.train_augs),
                label_encoder=self.label_encoder,
            )
            num_val_files = len(self.val_dataset)
            logging.info(f'Mode: val, number of files: {num_val_files}')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False,
        )
