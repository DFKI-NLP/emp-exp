import json
import os
from typing import Dict, Callable

import torch
from captum.attr import LayerIntegratedGradients, ShapleyValueSampling
from ignite.handlers import ModelCheckpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, AutoTokenizer, \
    XLNetForSequenceClassification

from train import get_dataset, EmpiricalExplainer
from utils_and_base_types import read_path, Configurable


class Explainer(Configurable):

    def validate_config(self, config: Dict) -> bool:
        raise NotImplementedError

    def from_config(cls, config: Dict):
        raise NotImplementedError

    def explain(self, batch):
        raise NotImplementedError

    def to(self, device):
        raise NotImplementedError


class ExplainerCaptum(Explainer):
    available_models = ['bert-base-cased', 'xlnet-base-cased']

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_inputs_and_additional_args(name_model: str, batch):
        assert name_model in ExplainerCaptum.available_models, f'Unkown model:  {name_model}'
        if name_model == 'bert-base-cased' or name_model == 'xlnet-base-cased':
            assert 'input_ids' in batch, f'Input ids expected for {name_model} but not found.'
            assert 'attention_mask' in batch, f'Attention mask expected for {name_model} but not found.'
            assert 'token_type_ids' in batch, f'Token type ids expected for model {name_model} but not found.'
            input_ids = batch['input_ids']
            additional_forward_args = (batch['attention_mask'], batch['token_type_ids'])
            return input_ids, additional_forward_args
        else:
            raise NotImplementedError

    @staticmethod
    def get_forward_func(name_model: str, model):
        assert name_model in ExplainerCaptum.available_models, f'Unkown model:  {name_model}'

        def bert_forward(input_ids, attention_mask, token_type_ids):
            input_model = {
                'input_ids': input_ids.long(),
                'attention_mask': attention_mask.long(),
                'token_type_ids': token_type_ids.long(),
            }
            output_model = model(**input_model)[0]
            return output_model

        if name_model == 'bert-base-cased' or name_model == 'xlnet-base-cased':
            return bert_forward
        else:  # when adding a model, also update ExplainerCaptum.available_models
            raise NotImplementedError(f'Unknown model {name_model}')

    def validate_config(self, config: Dict) -> bool:
        raise NotImplementedError

    def from_config(cls, config: Dict):
        raise NotImplementedError

    def explain(self, input):
        raise NotImplementedError


class ExplainerAutoModelInitializer(ExplainerCaptum):  # todo check if this is a mixin rather

    def __init__(self):
        super().__init__()
        self.name_model: str = None
        self.model: AutoModelForSequenceClassification = None
        self.path_model: str = None
        self.forward_func: Callable = None
        self.pad_token_id = None  #
        self.explainer = None
        self.device = None

    def validate_config(self, config: Dict) -> bool:
        assert 'name_model' in config, f'Provide the name of the model to explain. Available models: ' \
                                       f'{ExplainerCaptum.available_models}'
        assert 'path_model' in config, f'Provide a path to the model which should be explained.'
        # needed to deal w/ legacy code:
        assert 'mode_load' in config, f'Should the model be loaded using the ignite framework or huggingface?'
        assert 'num_labels' in config, f'Provide the number of labels.'
        return True

    @classmethod
    def from_config(cls, config):
        res = cls()
        res.validate_config(config)

        # model
        res.name_model = config['name_model']
        res.path_model = read_path(config['path_model'])
        res.mode_load = config['mode_load']
        if res.mode_load == 'huggingface':
            res.model = AutoModelForSequenceClassification.from_pretrained(res.path_model,
                                                                           num_labels=config['num_labels'])
        elif res.mode_load == 'ignite':
            res.model = AutoModelForSequenceClassification.from_pretrained(res.name_model, num_labels=config[
                'num_labels'])  # todo: num_labels hard coded for xlnet
            checkpoint = torch.load(res.path_model)
            to_load = {'model': res.model}
            ModelCheckpoint.load_objects(to_load=to_load,
                                         checkpoint=checkpoint)  # overwrite pretrained weights w/ fine-tuned weights
        else:
            raise NotImplementedError
        res.forward_func = res.get_forward_func(name_model=res.name_model, model=res.model)
        res.pad_token_id = AutoTokenizer.from_pretrained(res.name_model).pad_token_id
        return res

    def to(self, device):
        self.device = device
        self.model.to(self.device)

    def get_baseline(self, batch):
        if self.pad_token_id == 0:
            # all non-special token ids are replaced by 0, the pad id
            baseline = batch['input_ids'] * batch['special_tokens_mask']
            return baseline
        else:
            baseline = batch['input_ids'] * batch['special_tokens_mask']  # all input ids now 0
            # add pad_id everywhere,
            # subtract again where special tokens are, leaves non special tokens with pad id
            # and conserves original pad ids
            baseline = (baseline + self.pad_token_id) - (batch['special_tokens_mask'] * self.pad_token_id)
            return baseline

    def explain(self, input):
        raise NotImplementedError


class ExplainerLayerIntegratedGradients(ExplainerAutoModelInitializer):
    def __init__(self):
        super().__init__()
        self.name_layer: str = None
        self.layer = None
        self.n_samples: int = None
        self.internal_batch_size = None

    def validate_config(self, config: Dict) -> bool:
        super().validate_config(config)
        assert 'n_samples' in config, 'Define how many samples to take along the straight line path from the baseline.'
        assert 'internal_batch_size' in config, 'Define an internal batch size for the attribute method.'

    @staticmethod
    def get_embedding_layer_name(model):
        if isinstance(model, BertForSequenceClassification):
            return 'bert.embeddings'
        elif isinstance(model, XLNetForSequenceClassification):
            return 'transformer.word_embedding'
        else:
            raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        res = super().from_config(config)
        res.validate_config(config=config)
        res.n_samples = config['n_samples']
        res.internal_batch_size = config['internal_batch_size']
        res.name_layer = res.get_embedding_layer_name(res.model)
        for name, layer in res.model.named_modules():
            if name == res.name_layer:
                res.layer = layer
                break
        assert res.layer is not None, f'Layer {res.name_layer} not found.'
        res.explainer = LayerIntegratedGradients(forward_func=res.forward_func, layer=res.layer)
        return res

    def explain(self, batch):
        self.model.eval()
        self.model.zero_grad()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        inputs, additional_forward_args = self.get_inputs_and_additional_args(name_model=self.name_model, batch=batch)
        predictions = self.forward_func(inputs, *additional_forward_args)
        target = torch.argmax(predictions, dim=1)
        base_line = self.get_baseline(batch=batch)
        attributions = self.explainer.attribute(inputs=inputs,
                                                n_steps=self.n_samples,
                                                additional_forward_args=additional_forward_args,
                                                target=target,
                                                baselines=base_line,
                                                internal_batch_size=self.internal_batch_size)
        attributions = torch.sum(attributions, dim=2)
        if isinstance(self.model, XLNetForSequenceClassification):
            # for xlnet, attributions.shape = [seq_len, batch_dim]
            # but [batch_dim, seq_len] is assumed
            attributions = attributions.T
        return attributions, predictions  # xlnet: [130, 1]


class ExplainerShapleyValueSampling(ExplainerAutoModelInitializer):

    def __init__(self):
        super().__init__()
        self.n_samples = None

    def validate_config(self, config: Dict) -> bool:
        super().validate_config(config)
        assert 'n_samples' in config, 'Define how many samples to take along the straight line path from the baseline.'

    @classmethod
    def from_config(cls, config):
        res = super().from_config(config)
        res.n_samples = config['n_samples']
        res.explainer = ShapleyValueSampling(res.forward_func)  # KernelShap(forward_func=res.forward_func)
        return res

    def get_feature_mask(self, special_tokens_mask):
        feature_mask = torch.zeros_like(special_tokens_mask)
        counter = 1
        for idx, mask in enumerate(special_tokens_mask[0]):
            if mask != 1:
                feature_mask[0][idx] = counter
                counter = counter + 1
        return feature_mask

    def explain(self, batch):
        self.model.eval()
        assert len(batch['input_ids']) == 1, 'This implementation assumes that the batch size is 1.'
        batch = {k: v.to(self.device) for k, v in batch.items()}
        inputs, additional_forward_args = self.get_inputs_and_additional_args(name_model=self.name_model, batch=batch)
        predictions = self.forward_func(inputs, *additional_forward_args)
        target = torch.argmax(predictions, dim=1)
        base_line = self.get_baseline(batch=batch)
        feature_mask = self.get_feature_mask(batch['special_tokens_mask'])
        attributions = self.explainer.attribute(inputs=inputs,
                                                n_samples=self.n_samples,
                                                additional_forward_args=additional_forward_args,
                                                target=target,
                                                baselines=base_line,
                                                feature_mask=feature_mask)
        return attributions, predictions


class ExplainerEmpirical(Explainer): # todo: bad name, train.py  already contains a class called EmpiricalExplainer

    def __init__(self):
        super().__init__()
        self.explainer = None
        self.input_keys = None
        self.device = None

    @staticmethod
    def validate_config(config: Dict) -> bool:
        assert 'input_keys' in config, 'Expecting input keys for the encoder but found none.'
        assert 'path_model' in config, 'Expecting path to the trained generative explainer but found none.'
        # The keys for the GenerativeExplainer class should also be included, will be tested when GE is initialized
        # from_config
        return True

    @classmethod
    def from_config(cls, config: Dict):
        res = cls()
        res.validate_config(config=config)
        explainer = EmpiricalExplainer.from_config(config)
        checkpoint = torch.load(read_path(config['path_model']))
        to_load = {'model': explainer}
        ModelCheckpoint.load_objects(to_load=to_load,
                                     checkpoint=checkpoint)
        res.explainer = explainer
        res.input_keys = config['input_keys']
        return res

    def explain(self, batch):
        inputs = {k: v.to(self.device) for k, v in batch.items() if k in self.input_keys}
        attributions = self.explainer(inputs)
        return attributions, None

    def to(self, device):
        self.device = device
        self.explainer.to(device)


def get_explainer(name: str, config: Dict):
    if name == 'integrated-gradients':
        res = ExplainerLayerIntegratedGradients.from_config(config=config)
        return res
    if name == 'shapley-sampling':
        res = ExplainerShapleyValueSampling.from_config(config=config)
        return res
    if name == 'empirical-explainer':
        res = ExplainerEmpirical.from_config(config=config)
        return res
    else:
        raise NotImplementedError


def detach_to_list(t):
    return t.detach().cpu().numpy().tolist()


def run_explain(config: Dict, logger):
    logger.info(f'(Progress) Explaining w/ config \n{json.dumps(config, indent=2)}')

    config_explainer = config['explainer']
    config_dataset = config['dataset']
    batch_size = config['batch_size']
    path_out = read_path(config['path_out'])
    explainer = get_explainer(name=config_explainer['name'],
                              config=config_explainer['config'])
    path_model = config_explainer['config']['path_model']
    devise = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Operating on devise {devise}.')
    explainer.to(devise)

    dataset = get_dataset(name=config_dataset['name'],
                          config=config_dataset['config'])
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

    assert not os.path.isfile(path_out), f'File {path_out} already exists.'
    file_out = open(path_out, 'w+')

    for idx_batch, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        logger.info(f'(Progress) Processing batch {idx_batch} / instance {idx_batch * batch_size}')
        attributions, predictions = explainer.explain(batch)
        for idx_instance in range(len(batch['input_ids'])):
            idx_instance_running = (idx_batch * batch_size) + idx_instance
            ids = detach_to_list(batch['input_ids'][idx_instance])
            lbls = detach_to_list(batch['labels'][idx_instance])
            atts = detach_to_list(attributions[idx_instance])
            preds = detach_to_list(predictions[idx_instance]) if predictions is not None else None
            result = {'dataset': config_dataset,
                      'batch': idx_batch,
                      'instance': idx_instance,
                      'index_running': idx_instance_running,
                      'explainer': config_explainer,
                      'input_ids': ids,
                      'labels': lbls,
                      'attributions': atts,
                      'predictions': preds,
                      'path_model': path_model}
            file_out.write(json.dumps(result) + os.linesep)
    logger.info('(Progress) Terminated normally')
