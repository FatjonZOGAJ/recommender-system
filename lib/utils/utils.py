import argparse
import logging
import pprint
import random
import time
from pathlib import Path

import numpy as np
import torch

from lib.utils.config import config


def init(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    # overwrite config with commandline args
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, time_str = create_logger(config, args.cfg, None, 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description='Collaborative Filtering')
    parser.add_argument('--cpu',
                        help='run on cpu',
                        type=int,
                        default=-1)
    # TODO: add yaml file if necessary
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=False,
                        type=str,
                        default='default')

    return parser.parse_args()


def create_logger(cfg, cfg_name, time_str=None, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    if not time_str:
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    final_output_dir = root_output_dir / cfg.MODEL / time_str  # cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger, str(final_output_dir), time_str


def reset_config(config, args, valid=False):
    if args.cpu != -1:  # check for default values with -1
        config.CPU = bool(args.cpu)
