import argparse
import json
import logging

import torch
from pandas import np

from data import run_download
from evaluate import run_evaluate
from explain import run_explain, run_compute_convergence_curve
from train import run_train
from utils_and_base_types import read_config, get_logger, get_code_version
from visualize import run_visualize

if __name__ == '__main__':
    version_code = get_code_version()
    logger = get_logger(name='pipeline', level=logging.INFO)
    logger.info('(Progress) Jobs started')
    logger.info(f'(Config) Code version {version_code}')

    torch.manual_seed(123)
    np.random.seed(123)

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', help='Config file', default='')
    args = parser.parse_args()
    if args.c != '':
        config_file = read_config(args.c)
        mode = args.c.split('/')[-1].split('.')[0]
        if mode == 'train':
            logger = None
        locals()[f'run_{mode}'](config=config_file, logger=logger)
    else:
        path_config = 'configs/run_job.jsonnet'
        config = read_config(path_config)
        logger.info(f'(Config) \n{json.dumps(config, indent=2)}')

        if config['download']:
            path_config_download = config['path_config_download']
            config_download = read_config(path_config_download)
            run_download(config=config_download, logger=logger)

        if config['train']:
            path_config_training = config['path_config_training']
            config_training = read_config(path_config_training)
            run_train(config=config_training)

        if config['explain']:
            path_config_explain = config['path_config_explain']
            config_explain = read_config(path_config_explain)
            run_explain(config=config_explain, logger=logger)

        if config['compute_convergence_curve']:
            path_config_compute_convergence_curve = config['path_config_compute_convergence_curve']
            config_compute_convergence_curve = read_config(path_config_compute_convergence_curve)
            run_compute_convergence_curve(config=config_compute_convergence_curve, logger=logger)

        if config['evaluate']:
            path_config_evaluate = config['path_config_evaluate']
            config_evaluate = read_config(path_config_evaluate)
            run_evaluate(config=config_evaluate, logger=logger)

        if config['visualize']:
            path_config_visualize = config['path_config_visualize']
            config_visualize = read_config(path_config_visualize)
            run_visualize(config=config_visualize, logger=logger)

    if logger:
        logger.info(f'(Progress) Jobs done')
    else:
        print('(Progress) Jobs done')
