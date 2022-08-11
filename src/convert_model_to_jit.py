import argparse
import logging
import os
import typing as tp
from runpy import run_path

import torch

from src.configs.base_config import Config
from src.model.model import MultiLabelClassifier


def parse() -> tp.Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', required=True, type=str, help='Path to experiment config file (*.py)',
    )
    parser.add_argument(
        '--ckpt_path', required=True, type=str, help='Path to experiment checkpoint (*.ckpt)',
    )
    parser.add_argument(
        '--path_to_save', type=str, default='../weights', help='Path to directory where .pt model will be saved',
    )
    parser.add_argument(
        '--check', action='store_true', help='Check correctness of converting by shape of output',
    )
    return parser.parse_args()


def convert_from_checkpoint(args: tp.Any, config: Config):
    model = MultiLabelClassifier.load_from_checkpoint(args.ckpt_path, criterion=config.train.criterion.loss)
    model_name = os.path.split(args.ckpt_path)[-1].replace('ckpt', 'pt')
    model_path = os.path.join(args.path_to_save, model_name)
    model.to_torchscript(file_path=model_path)

    if os.path.isfile(model_path):
        logging.info(f'Model was successfully saved. File name: {model_path}')
    else:
        logging.error('An error was occurred. Check paths and try again.')
    return model_path


def check(model_path: str, config: Config):
    model = torch.jit.load(model_path, map_location='cpu')
    nc = config.dataset.num_channels
    img_size = config.dataset.img_size

    dummy_input = torch.randn([1, nc, img_size, img_size])
    if model(dummy_input).shape != (1, config.dataset.num_classes):
        raise AssertionError('Output shape is not corrected')

    dummy_input = torch.randn([10, nc, img_size, img_size])
    if model(dummy_input).shape != (10, config.dataset.num_classes):
        raise AssertionError('Output shape is not corrected')

    logging.info('Model can be loaded and outputs look good!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse()
    config_module = run_path(args.config)
    exp_config = config_module['CONFIG']
    pt_model_path = convert_from_checkpoint(args, exp_config)
    if args.check:
        check(pt_model_path, exp_config)
