import copy
import json
import os
from typing import Dict

from sklearn.metrics import f1_score
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from data import get_dataset


def run_evaluate(config: Dict, logger, verbose=False):
    logger.info(f'(Progress) Evaluating w/ config \n{json.dumps(config, indent=2)}')

    name_target = config['dataset_target']['name']
    config_target = config['dataset_target']['config']
    dataset_target = get_dataset(name=name_target, config=config_target)
    dataloader_target = DataLoader(dataset=dataset_target, batch_size=1)

    if config['compute_convergence']:
        mses_avg_per_file = []
        for file_idx, filename in enumerate(config['filenames']):

            name_approximation = copy.copy(name_target)
            config_approximation = copy.copy(config_target)
            config_approximation['paths_json_files'] = filename
            dataset_approximation = get_dataset(name=name_approximation, config=config_approximation)
            dataloader_approximation = DataLoader(dataset=dataset_approximation, batch_size=1)

            mses_acc = []
            for idx, (target, approximation) in tqdm(enumerate(zip(dataloader_target, dataloader_approximation))):
                assert torch.all(target['input_ids'] == approximation['input_ids'])
                if verbose:
                    logger.info('(Progress) Input_ids identical.')

                attributions_approximation = approximation['attributions']
                attributions_target = target['attributions']
                mse = mse_loss(input=attributions_approximation, target=attributions_target)
                mses_acc.append(mse)

            logger.info(f'### File {filename} ###')

            mses_sum = sum(mses_acc)
            logger.info(f'(Progress) Sum of MSEs: {mses_sum}')

            mses_avg = mses_sum / len(mses_acc)
            logger.info(f'(Progress) Average MSE: {mses_avg}')

            mses_avg_per_file.append((filename, str(mses_avg.item())))

        logger.info("--- Summary ---")
        logger.info(f"Target: {config_target['paths_json_files']}")
        for filename, score in mses_avg_per_file:
            logger.info(f'{score} / {filename}')

        logger.info(f'(Progress) Copy friendly summary: \n{os.linesep.join([score for _, score in mses_avg_per_file])}')

    if config['compute_f1']:
        if 'labels' not in config_target:  # not needed for convergence curves
            logger.info("(Config) Labels not found in columns of interest, adding labels.")
            config_target['columns'].append('labels')
        if 'predictions' not in config_target:
            logger.info("(Config) Predictions not found in columns of interest, adding predictions.")
            config_target['columns'].append('predictions')
        logger.info(f"(Progress) Computing f1 score w/ config \n{json.dumps(config, indent=2)}")
        dataset_target = get_dataset(name=name_target, config=config_target)
        dataloader_target = DataLoader(dataset=dataset_target, batch_size=1)
        predictions = []
        labels = []
        for instance in tqdm(dataloader_target, desc='Instances'):
            predictions.append(instance['predictions'])
            labels.append(instance['labels'])
        predictions = [torch.argmax(logits).item() for logits in predictions]
        labels = [label.item() for label in labels]
        f1 = f1_score(y_true=labels, y_pred=predictions, average='weighted')
        logger.info(f'f_score: {f1}')
