import pytorch_lightning as pl
import torch

from src.configs.base_config import LRScheduler, Optimizer
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
        lr_scheduler=None,
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
