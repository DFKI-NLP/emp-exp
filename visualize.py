import json
import math
import os

from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from data import get_dataset
from utils_and_base_types import read_path, detach_to_list

import numpy as np


class RGB:
    def __init__(self, red, green, blue, score):
        self.red = red
        self.green = green
        self.blue = blue
        self.score = round(score, ndigits=3) if score is not None else score

    def __str__(self):
        return 'rgb({},{},{})'.format(self.red, self.green, self.blue)


class Sequence:
    def __init__(self, words, scores):
        assert (len(words) == len(scores))
        self.words = words
        self.scores = scores
        self.size = len(words)

    def words_rgb(self, gamma=1.0, token_pad=None, position_pad='back'):
        rgbs = list(map(lambda tup: self.rgb(word=tup[0], score=tup[1], gamma=gamma), zip(self.words, self.scores)))
        if token_pad is not None:
            if token_pad in self.words:
                if position_pad == 'back':
                    return zip(self.words[:self.words.index(token_pad)], rgbs)
                elif position_pad == 'front':
                    first_token_index = list(reversed(self.words)).index(token_pad)
                    return zip(self.words[-first_token_index:], rgbs[-first_token_index:])
                else:
                    return NotImplementedError
        return zip(self.words, rgbs)

    def compute_length_without_pad_tokens(self, special_tokens: List[str]):
        counter = 0
        for word in self.words:
            if word not in special_tokens:
                counter = counter + 1
        return counter

    @staticmethod
    def gamma_correction(score, gamma):
        return np.sign(score) * np.power(np.abs(score), gamma)

    def rgb(self, word, score, gamma, threshold=0):
        assert not math.isnan(score), 'Score of word {} is NaN'.format(word)
        score = self.gamma_correction(score, gamma)
        if score >= threshold:
            r = str(int(255))
            g = str(int(255 * (1 - score)))
            b = str(int(255 * (1 - score)))
        else:
            b = str(int(255))
            r = str(int(255 * (1 + score)))
            g = str(int(255 * (1 + score)))
        return RGB(r, g, b, score)


def token_to_html(token, rgb):
    return f"<span style=\"background-color: {rgb}\"> {token.replace('<', '').replace('>', '')} </span>"


def summarize(summary: Dict):
    res = "<h4>"
    for k, v in summary.items():
        res += f"{k}: {summary[k]} <br/>"
    res += "</h4>"
    return res


def run_visualize(config: Dict, logger):
    logger.info("(Progress) Generating visualizations")
    logger.info(f"(Config) Received config \n{json.dumps(config, indent=2)}")
    tokenizer = AutoTokenizer.from_pretrained(config['name_model'])
    datasets = []
    for dct in config['datasets']:
        dataset = get_dataset(name=dct['name'], config=dct['config'])
        datasets.append(dataset)
    file_out = open(read_path(config['path_file_out']), 'w+')
    number_datasets = len(datasets)
    number_instances = len(datasets[0])
    for index_instance in tqdm(range(number_instances), desc='Instances'):
        html = f"<html><h3>"
        html += f"<h2>Instance: {index_instance} | Dataset: {config['name_dataset_html']} | Model: {config['name_model_html']}"
        html += '</h3><div style=\"border:3px solid #000;\">'
        for index_dataloader in range(number_datasets):
            html += "<div>"
            if index_dataloader > 0:
                html += "<hr>"
            dataloader = datasets[index_dataloader]
            instance = dataloader[index_instance]
            tokens = [tokenizer.decode(token_ids=token_id) for token_id in instance['input_ids']]
            atts = detach_to_list(instance['attributions'])
            if config['normalize']:
                max_abs_score = max(max(atts), abs(min(atts)))
                atts = [(score / max_abs_score) for score in atts]
            sequence = Sequence(words=tokens, scores=atts)
            words_rgb = sequence.words_rgb(token_pad=tokenizer.pad_token,
                                           position_pad=config['position_pad'])  # todo: <pad> is xlnet pad token
            config_dataset = config['datasets'][index_dataloader]
            summary = {}
            if config_dataset['name_explainer'] == 'shapley-sampling':
                number_of_non_special_tokens = sequence.compute_length_without_pad_tokens(
                    special_tokens=tokenizer.all_special_tokens)
                summary['Explainer'] = f"Shapley Value Sampling, {config_dataset['n_samples']} samples"
                summary['Required Passes'] = number_of_non_special_tokens * config_dataset['n_samples']
            elif config_dataset['name_explainer'] == 'integrated-gradients':
                summary['Explainer'] = f"Integrated Gradients, {config_dataset['n_samples']} samples"
                summary['Required Passes'] = config['datasets'][index_dataloader]['n_samples'] * 2
            elif config_dataset['name_explainer'] == 'empirical':
                summary['Explainer'] = "Empirical"
                summary['Required Passes'] = 1
            else:
                raise NotImplementedError
            if 'labels' in instance:
                summary['True Label Index'] = str(detach_to_list(instance['labels']))
                summary['True Label'] = f"{config_dataset['labels'][str(detach_to_list(instance['labels']))]}"
            if 'predictions' in instance:
                summary['Logits'] = detach_to_list(instance['predictions'])
                summary[
                    'Predicted Label'] = f"{config_dataset['labels'][str(detach_to_list(torch.argmax(instance['predictions'])))]}"
            html += summarize(summary)

            for word, rgb in words_rgb:  # brackets to reuse iterator
                html += token_to_html(word, rgb)
            html += "</br></br>"
            html += "</div>"
        html += "</div></br></br></br></html>"
        file_out.write(html + os.linesep)
