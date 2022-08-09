from clearml import Task
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from src.configs.config import Config
from src.configs.config import CONFIG as config
from src.data.dataset import AmazonDataModule


def main(config: Config):
    # model
    model = config.model.module(
        optimizer=config.train.optimizer,
        lr_scheduler=config.train.lr_scheduler,
        criterion=config.train.criterion,
        **config.model.params
    )

    # data module
    datamodule = AmazonDataModule(
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
    task = Task.init(project_name="examples", task_name="PyTorch lightning MNIST example")

    # trainer
    callbacks = list(config.train.callbacks.__dict__.values())
    callbacks = filter(lambda x: x is not None, callbacks)
    trainer = Trainer(
        callbacks=[
            TQDMProgressBar(refresh_rate=1),
            RichModelSummary(),
            *callbacks
        ],
        **config.train.trainer_params
    )

    if config.train.trainer_params['auto_scale_batch_size'] is not None or\
            config.train.trainer_params['auto_lr_find'] is not None:
        trainer.tune(model=model, datamodule=datamodule)

    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=config.train.ckpt_path,
    )


if __name__ == '__main__':
    seed_everything(config.common.seed, workers=True)
    main(config)
