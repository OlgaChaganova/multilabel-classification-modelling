import typing as tp

from torchvision import transforms as tt

AUGMENTATION_MODES = tp.Literal[
    'default'
]


NORMALIZE = tt.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def get_train_aug(mode: str, img_size: int):
    if mode == 'default':
        train_augs = tt.Compose([
            tt.Resize((img_size, img_size)),
            tt.RandomHorizontalFlip(),
            tt.RandomVerticalFlip(),
            tt.ToTensor(),
            NORMALIZE,
        ])

    else:
        augmentations_mode = tp.get_args(AUGMENTATION_MODES)
        raise ValueError(
            f'Unknown mode of train augmentations: {mode}. Available modes are: {augmentations_mode}',
        )
    return train_augs


def get_val_aug(mode: str, img_size: int):
    if mode == 'default':
        val_augs = tt.Compose([
            tt.Resize((img_size, img_size)),
            tt.ToTensor(),
            NORMALIZE,
        ])
    else:
        augmentations_mode = tp.get_args(AUGMENTATION_MODES)
        raise ValueError(
            f'Unknown mode of valid augmentations: {mode}. Available modes are: {augmentations_mode}',
        )
    return val_augs
