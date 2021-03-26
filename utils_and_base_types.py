import linecache
import logging
import math
import os
import subprocess
import warnings
from os.path import expanduser

__author__ = 'robert.schwarzenberg@dfki.de'

from datetime import datetime
import random
from typing import List, Dict

import json
import _jsonnet
import torch


class Configurable:

    def __init__(self):
        """A  DataBroker is an abstract class for down loaders, tokenizers and other components that handle download,
        convert and write Datapoints.
        """
        pass

    def validate_config(self, config: Dict) -> bool:
        """Validate a config file. Is true if all required fields to configure this downloader are present.
        :param config: The configuration file to validate.
        :returns: True if all required fields exist, else False.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Dict):
        """Initializes the Preprocessor from a config file. The required fields in the config file are validated in
        validate_config.
        :param config: The config file to initialize this Downloader from.
        :return: The configured Downloader.
        """
        res = cls()
        res.validate_config(config)
        for k, v in config.items():
            assert k in res.__dict__, f'Unknown key: {k}'
            setattr(res, k, v)
        return res


def get_logger(name: str, file_out: str = None, level: int = None):
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)

    # Create handlers
    c_handler = logging.StreamHandler()
    if level is not None:
        c_handler.setLevel(level)
    c_format = logging.Formatter('%(asctime)s -%(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    if file_out is not None:
        f_handler = logging.FileHandler(file_out, mode='w+')
        if level is not None:
            f_handler.setLevel(level)
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    return logger


def value_error(message: str, logger: logging.Logger):
    logger.error(message)
    raise ValueError(message)


def get_code_version():
    result = 'No version available.'
    try:
        process = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
        result = process.communicate()[0].strip()
    except:
        pass
    return result


def get_time():
    now = datetime.now()
    now = now.strftime("%Y-%m-%d-%H-%M-%S")
    return now


class PredExplanation:
    def __init__(self, tokens, pred_scores, scores):
        self.tokens = tokens
        self.pred_scores = pred_scores
        self.scores = scores

    def __repr__(self):
        return json.dumps(self.__dict__)

    def __str__(self):
        return self.__repr__()

    @classmethod
    def from_dct(cls, dct):
        res = PredExplanation(
            tokens=dct['tokens'],
            scores=dct['scores'],
            pred_scores=dct['pred_scores'],
        )
        return res


def read_config(path) -> Dict:
    config = json.loads(_jsonnet.evaluate_file(path))
    return config


def create_or_fail(path, logger):
    try:
        file = open(path, 'a+')
    except:
        logger.error(f'Problems opening file: {path}')
    file.close()


def get_logger_and_path(config):
    """Extracted from main for readability."""
    path_explanation_dir = read_path(config['explanation']['config']['path_dir_target'])
    assert os.path.isdir(path_explanation_dir), f'Not a directory (path_explanation_dir): {path_explanation_dir}'
    _now = get_time()
    path_log = os.path.join(path_explanation_dir, _now + '_explanation.log')
    logger = get_logger(name='explain', level=logging.INFO, file_out=path_log)
    logger.info("Config\n")
    logger.info(json.dumps(config, indent=2))

    path_explanation = os.path.join(path_explanation_dir, _now + '_explantion.jsonl')
    create_or_fail(path_log, logger=logger)
    create_or_fail(path_explanation, logger=logger)
    logger.info(f'(Config) Path log: {path_log}')
    logger.info(f'(Config) Path explanation: {path_explanation}')
    return logger, path_explanation


def read_path(path):
    """Replaces $HOME w/ home directory."""
    home = expanduser("~")
    return path.replace("$HOME", home)


class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True


def align_splits_by_id(path_source_split, path_target_split, path_out):
    """Collects the ids of the json lines from the source split, then filters the target split for the ids and writes
    them to path_out"""

    ids_source_split = []
    with open(path_source_split, 'r+') as file_source:
        for line in file_source.readlines():
            jsnl = json.loads(line)
            id = jsnl['id']
            ids_source_split.append(id)

    ids_target_split = set()
    with open(path_target_split, 'r+') as file_target:
        with open(path_out, 'a+') as file_out:
            for line in file_target.readlines():
                jsnl = json.loads(line.strip())
                id = jsnl['id']
                if id in ids_source_split:
                    file_out.write(json.dumps(jsnl) + os.linesep)
                    ids_target_split.add(id)
    assert len(ids_target_split) == len(ids_source_split), 'The target and source ids should be equal.'


def write_splits(path_in: str, percentage_of_train_split: float= 80):
    """Takes a file, randomizes it, splits it into train, dev, test splits,
    conditioned on the number_train_expamples. Returns paths of train, dev, test splits."""
    path_wo_extension = os.path.splitext(path_in)

    path_train_split = path_wo_extension[0] + '.train' + path_wo_extension[1]
    path_dev_split = path_wo_extension[0] + '.dev' + path_wo_extension[1]
    path_test_split = path_wo_extension[0] + '.test' + path_wo_extension[1]

    line_counter = 0
    with open(path_in, 'r+') as fin:
        for idx, _ in enumerate(fin.readlines()):
            line_counter = line_counter + 1

    number_train_examples = math.ceil((percentage_of_train_split/100) * line_counter)

    assert line_counter > number_train_examples, f'Sanity check failed: {line_counter} <= {number_train_examples}'

    indices = list(range(0, line_counter))
    random.shuffle(indices)

    train_indices = indices[:number_train_examples]

    dev_size = int((line_counter - number_train_examples) / 2)
    dev_indices = indices[number_train_examples:number_train_examples + dev_size]

    test_indices = indices[number_train_examples + dev_size:]

    def transfer(indices, path_from, path_to):
        with open(path_to, 'w+') as fout:
            counter = 0
            for index in indices:
                line = linecache.getline(path_from, index).strip()
                if len(line) == 0:
                    warnings.warn('Encountered empty line.')
                    continue
                if counter == 0:
                    fout.write(line)
                else:
                    fout.write(os.linesep + line)
                counter = counter + 1

    transfer(path_from=path_in, path_to=path_train_split, indices=train_indices)
    transfer(path_from=path_in, path_to=path_dev_split, indices=dev_indices)
    transfer(path_from=path_in, path_to=path_test_split, indices=test_indices)

    return path_train_split, path_dev_split, path_test_split


def detach_to_list(t: torch.Tensor):
    return t.detach().cpu().numpy().tolist()




