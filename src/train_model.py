import argparse
import os
import typing as tp
from runpy import run_path

import numpy as np
from clearml import Task
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from src.configs.base_config import Config
from src.data.dataset import AmazonDataModule
from src.model.model import MultiLabelClassifier


def parse() -> tp.Any:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Path to experiment config file (*.py)')
    return parser.parse_args()


def main(args: tp.Any, config: Config):
    # model
    model = MultiLabelClassifier(
        optimizer=config.train.optimizer,
        lr_scheduler=config.train.lr_scheduler,
        criterion=config.train.criterion.loss,
        **config.model.model_params,
    )

    # data module
    datamodule = AmazonDataModule(
        imlist_filename=config.dataset.imlist_filename,
        root=config.dataset.root,
        batch_size=config.dataset.batch_size,
        img_type=config.dataset.img_type,
        img_size=config.dataset.img_size,
        test_size=config.dataset.test_size,
        train_aug_mode=config.dataset.train_augmentations,
        valid_aug_mode=config.dataset.valid_augmentations,
        num_workers=config.dataset.num_workers,
    )

    # clearml task
    task = Task.init(project_name=config.project.project_name, task_name=config.project.task_name)

    # save config.py for reproducibility
    task.upload_artifact('exp_config', artifact_object=args.config, delete_after_upload=False)

    # save label_encoder classes for future inference
    label_encoder_filename = os.path.join(os.getcwd(), 'label_encoder_classes.npy')
    np.save(label_encoder_filename, datamodule.label_encoder.classes_)
    task.upload_artifact('label_encoder_classes', artifact_object=label_encoder_filename, delete_after_upload=False)

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
        **trainer_params,
    )

    if trainer_params['auto_scale_batch_size'] is not None or trainer_params['auto_lr_find'] is not None:
        trainer.tune(model=model, datamodule=datamodule)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=config.train.ckpt_path,
    )


if __name__ == '__main__':
    args = parse()
    config_module = run_path(args.config)
    exp_config = config_module['CONFIG']
    seed_everything(exp_config.common.seed, workers=True)
    main(args, exp_config)
