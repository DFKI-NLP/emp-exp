import json
from typing import Dict

from datasets import load_dataset, load_from_disk, Features, Value
from transformers import AutoTokenizer

from utils_and_base_types import read_path


def run_download(config: Dict, logger):
    logger.info(f'(Progress) Download invoked with config \n{json.dumps(config, indent=2)}')
    dataset = load_dataset(config['name'], split=config['split'])
    dataset.save_to_disk(dataset_path=read_path(config['path_out']))
    logger.info(f'(Progress) Terminated normally')


def get_dataset(name: str, config: Dict):
    """Returns a pytorch dataset."""

    if name == 'huggingface.imdb' or name == 'huggingface.snli':
        tokenizer = AutoTokenizer.from_pretrained(config['name_model'])
        dataset = load_from_disk(read_path(config['path_dataset']))

        start = 0 if 'start' not in config else config['start']
        if start < 0:
            start = 0

        end = len(dataset) if 'end' not in config else config['end']
        if end < 0:
            end = len(dataset)

        dataset = dataset.select(indices=range(start, end))

        def encode_snli(instances):
            return tokenizer(instances['premise'],
                             instances['hypothesis'],
                             truncation=True,
                             padding='max_length',
                             max_length=config['max_length'],
                             return_special_tokens_mask=True)

        def encode_imdb(instances):
            return tokenizer(instances['text'],
                             truncation=True,
                             padding='max_length',
                             max_length=config['max_length'],
                             return_special_tokens_mask=True)

        if name == 'huggingface.imdb':
            dataset = dataset.map(encode_imdb, batched=True, batch_size=config['batch_size'])
        elif name == 'huggingface.snli':
            dataset = dataset.map(encode_snli, batched=True, batch_size=config['batch_size'])
        dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True,
                              batch_size=config['batch_size'])
        if name == 'huggingface.snli':
            dataset = dataset.filter(lambda examples: examples['labels'] != -1)  # -1 if ground truth unknown in snli
        dataset.set_format(type='torch', columns=config['columns'])
        return dataset

    if name == 'local.explanations':
        def encode_local(instances):
            res = {k: instances[k] for k in config['columns']}
            return res

        paths_json_files = config['paths_json_files']
        if isinstance(paths_json_files, list):
            paths_json_files = [read_path(path) for path in paths_json_files]
        else:
            paths_json_files = read_path(paths_json_files)

        dataset = load_dataset('json', data_files=paths_json_files)
        dataset = dataset['train']  # todo: why is this necessary, why is a Dict returned?
        features = dataset.features.copy()
        # type handling
        features['input_ids'].feature = Value(dtype='int64')
        features['attributions'].feature = Value(dtype='float32')
        features = Features(features)
        dataset.cast_(features)

        dataset = dataset.map(encode_local, batched=True)
        dataset.set_format(type='torch', columns=config['columns'])
        return dataset  # todo: why is the train-indexing this necessary?
    else:
        raise NotImplementedError
