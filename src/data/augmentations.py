import typing as tp
import torchvision.transforms as T


AUGMENTATION_MODES = tp.Literal[
    'default'
]


NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])


def get_train_aug(mode: str, img_size: int):
    if mode == 'default':
        train_augs = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            NORMALIZE
        ])

    else:
        raise ValueError(
            f'Unknown mode of train augmentations: {mode}. Available modes are: {tp.get_args(AUGMENTATION_MODES)}'
        )
    return train_augs


def get_val_aug(mode: str, img_size: int):
    if mode == 'default':
        val_augs = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            NORMALIZE
        ])
    else:
        raise ValueError(
            f'Unknown mode of valid augmentations: {mode}. Available modes are: {tp.get_args(AUGMENTATION_MODES)}'
        )
    return val_augs
