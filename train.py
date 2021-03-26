import json
import logging
import os
from typing import Dict, Union
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from overrides import overrides
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoModel, BertForSequenceClassification, \
    AutoTokenizer, XLNetForSequenceClassification

from data import get_dataset
from utils_and_base_types import read_config, read_path, get_logger, get_time, Configurable
from ignite.metrics import Metric, MeanSquaredError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import numpy as np


# metrics
class F1(Metric):
    """Custom implementation of the F1 metric."""

    def __init__(self, average='weighted', output_transform=lambda x: x):
        self._y_pred = None
        self._y_true = None
        self.average = average
        super(F1, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._y_pred = None
        self._y_true = None
        super(F1, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        y_pred_max_indices = torch.argmax(y_pred, dim=1)

        if self._y_pred is None:
            self._y_true = y
            self._y_pred = y_pred_max_indices
        else:
            self._y_true = torch.cat((self._y_true, y))
            self._y_pred = torch.cat((self._y_pred, y_pred_max_indices))

    @sync_all_reduce("_y_pred", "_y_true")
    def compute(self):
        res = f1_score(y_pred=self._y_pred.cpu(), y_true=self._y_true.cpu(), average=self.average)
        return res


class NegativeMSE(MeanSquaredError):
    """Negative Mean Squared Error. This metric is used to judge the quality of a generative explainer during
    train."""

    def __init__(self):
        super().__init__()

    @sync_all_reduce("_sum_of_squared_errors", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        res = super().compute()
        return -res


# models
class EmpiricalExplainer(torch.nn.Module, Configurable):

    def __init__(self):
        """An Empirical Explainer learns from saliency data to approximate an expensive explainer. The architecture of
        this explainer is as follows: The explainer copies the encoder of the downstream model and puts a fully
        connected layer on top. The fully connected layer and the encoder are then fine-tuned towards predicting the
        saliency scores of an expensive target explainer."""
        super().__init__()
        self.name_encoder = None
        self.seq_len = None
        self.dim_embedding = None
        self.encoder = None
        self.decoder = None
        self.pad_token_id = None
        self.attend_on_pad_tokens = None

    def get_encoder(self, model):
        """Retrieves the encoder from a downstream model. Please note that this is highly dependant on the internals of
        the downstream model. We implemented this using transformers package version 4.1.1.s"""
        if isinstance(model, BertForSequenceClassification):
            return model.bert
        if isinstance(model, XLNetForSequenceClassification):
            return model.transformer  # todo is this right?
        else:
            raise NotImplementedError

    @overrides
    def validate_config(self, config: Dict) -> bool:
        assert 'name_encoder' in config, 'Define encoder name.'
        assert 'seq_len' in config, 'Define sequence length.'
        assert 'dim_embedding' in config, 'Define embedding dimension.'
        return True

    @classmethod
    def from_config(cls, config: Dict):
        res = cls()
        res.name_encoder = config['name_encoder']
        res.pad_token_id = AutoTokenizer.from_pretrained(res.name_encoder).pad_token_id
        res.attend_on_pad_tokens = config['attend_on_pad_tokens']
        res.seq_len = config['seq_len']
        res.dim_embedding = config['dim_embedding']
        if 'downstream' in config and config['downstream'] is not None:
            mode_load = config['mode_load']
            path = read_path(config['downstream'])
            if mode_load == 'ignite':
                checkpoint = torch.load(path)
                downstream = AutoModelForSequenceClassification.from_pretrained(res.name_encoder,
                                                                                num_labels=config['num_labels'])
                to_load = {'model': downstream}
                ModelCheckpoint.load_objects(to_load=to_load,
                                             checkpoint=checkpoint)
                res.encoder = res.get_encoder(downstream)
            elif mode_load == 'huggingface':
                downstream = AutoModelForSequenceClassification.from_pretrained(path)
                res.encoder = res.get_encoder(downstream)
        else:  # when loaded for inference, load pretrained model then overwrite with fine-tuned generative explainer
            res.encoder = AutoModel.from_pretrained(res.name_encoder)  # AutoModel.from_pretrained(res.name_encoder)
        res.decoder = torch.nn.Linear(res.seq_len * res.dim_embedding, res.seq_len)
        return res

    def forward(self, batch):
        # handle attention on pad tokens, possibly overwrites attention_mask returned by HF tokenizer
        if not self.attend_on_pad_tokens:
            attention_mask = torch.ones_like(batch['input_ids']).to(self.encoder.device)
            attention_mask = (attention_mask * batch['input_ids'] != self.pad_token_id).type(torch.uint8)
            batch['attention_mask'] = attention_mask
        elif self.attend_on_pad_tokens: # overwrite HF attention mask
            attention_mask = torch.ones_like(batch['input_ids']).to(self.encoder.device)
            batch['attention_mask'] = attention_mask
        encoding = self.encoder(**batch)[0]
        result = self.decoder(encoding.view((-1, self.seq_len * self.dim_embedding)))
        return result


def get_model(name_model: str, config: Dict = None):
    """Get a model, specified by name."""
    if name_model == 'bert-base-cased' or name_model == 'xlnet-base-cased':
        model = AutoModelForSequenceClassification.from_pretrained(name_model, num_labels=config[
            'num_labels'])  # todo num_labels was fixed w/o testing train.py
        return model
    if name_model == 'empirical-explainer':
        return EmpiricalExplainer.from_config(config=config)
    else:
        raise NotImplementedError


def get_metric(name_model: str):
    """Get the appropriate metrics to evaluate the model on, specified by model name.
    Returns the metric and its name."""
    if name_model == 'bert-base-cased' or name_model == 'xlnet-base-cased':
        return F1(), 'f1'
    if name_model == 'empirical-explainer':
        return NegativeMSE(), 'nmse'
    else:
        raise NotImplementedError


def run_train(config: Dict, logger=None):
    torch.manual_seed(123)
    np.random.seed(123)

    def get_train_step(name_model: str, model, optimizer):
        """Get the appropriate train step for the model, which is specified by name."""

        # trainer
        def train_step_downstream_transformer(engine, batch):  # todo: rename; this forward applies to bert + xlnet
            model.train()
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            return loss.item()

        def train_step_empirical_explainer(engine, batch):
            model.train()
            optimizer.zero_grad()
            target = batch['attributions'].to(device)  # attributions are used as target
            inputs = {'input_ids': batch['input_ids'].to(device)}
            output = model(inputs)
            criterion = torch.nn.MSELoss()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            return loss.item()

        if name_model == 'bert-base-cased':
            return train_step_downstream_transformer
        if name_model == 'xlnet-base-cased':
            return train_step_downstream_transformer  # not a bug, the inputs are the same
        elif name_model == 'empirical-explainer':
            return train_step_empirical_explainer
        else:
            raise NotImplementedError(f'Unknown model: {name_model}')

    def get_val_step(name_model: str, model):
        """Get the appropriate validation step for the model which is specified by name."""

        def validation_step_downstream_bert(engine, batch):  # todo: rename this applies to bert + xlnet
            model.eval()
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                y = batch['labels']
                y_pred = model(**batch).logits
                return y_pred, y

        def validation_step_empirical_explainer(engine, batch):
            model.eval()
            with torch.no_grad():
                inputs = {'input_ids': batch['input_ids'].to(device)} # {k: v.to(device) for k, v in batch.items() if k != 'attribution'}
                y = batch['attributions'].to(device)
                y_pred = model(inputs)
                return y_pred, y

        if name_model == 'bert-base-cased':
            return validation_step_downstream_bert
        if name_model == 'xlnet-base-cased':
            return validation_step_downstream_bert  # same input keys as for bert
        elif name_model == 'empirical-explainer':
            return validation_step_empirical_explainer
        else:
            raise NotImplementedError

    # config
    name_model = config['model']['name']
    config_model = config['model']['config']
    batch_size_train = config['batch_size_train']
    batch_size_validation = config['batch_size_validation']
    name_dataset = config['dataset_train']['name']
    _now = get_time()
    path_dir_model = read_path(config['path_dir_model'])
    prefix_model = f'{_now}.{name_model}.{name_dataset}'
    logging_training_loss_every = config['logging_training_loss_every']
    max_epochs = config['max_epochs']
    patience = config['patience']
    name_dataset_reader_train = config['dataset_train']['name']
    config_dataset_reader_train = config['dataset_train']['config']
    name_dataset_reader_test = config['dataset_validation']['name']
    config_dataset_reader_test = config['dataset_validation']['config']

    # logger
    if logger is None:
        logger = get_logger(name=f'Training', level=logging.INFO)
    logger.info(f'(Config)\n {json.dumps(config, indent=2)}\n')
    logger.info(f'(Config) Writing checkpoints to: {os.path.join(path_dir_model, prefix_model)}')

    # model
    model = get_model(name_model=name_model, config=config_model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # data loaders
    dataset_reader_train = get_dataset(name=name_dataset_reader_train,
                                       config=config_dataset_reader_train)
    loader_train = DataLoader(dataset=dataset_reader_train,
                              batch_size=batch_size_train,
                              shuffle=True)
    dataset_reader_validation = get_dataset(name=name_dataset_reader_test, config=config_dataset_reader_test)
    loader_validation = DataLoader(dataset=dataset_reader_validation,
                                   batch_size=batch_size_validation,
                                   shuffle=False)

    # optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)

    # trainer and validator
    train_step = get_train_step(name_model=name_model, model=model, optimizer=optimizer)
    trainer = Engine(train_step)

    validation_step = get_val_step(name_model=name_model, model=model)
    evaluator = Engine(validation_step)

    metric, name_metric = get_metric(name_model=name_model)
    metric.attach(evaluator, name_metric)

    # handlers: early stopping and model checkpoint
    handler_early_stopping = EarlyStopping(patience=patience,
                                           score_function=lambda engine: engine.state.metrics[name_metric],
                                           trainer=trainer)
    handler_early_stopping.logger.level = logging.INFO
    evaluator.add_event_handler(Events.COMPLETED, handler_early_stopping)

    handler_model_checkpoint = ModelCheckpoint(dirname=read_path(path_dir_model),  # todo hard coded
                                               score_function=lambda engine: engine.state.metrics[name_metric],
                                               score_name=name_metric,
                                               filename_prefix=f'{prefix_model}',
                                               n_saved=1,
                                               create_dir=False,
                                               require_empty=False)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler_model_checkpoint, to_save={'model': model})

    pbar = ProgressBar()
    pbar.attach(evaluator)
    pbar.attach(trainer, ['loss']) # todo loss is misssing

    # logging
    @trainer.on(Events.ITERATION_COMPLETED(every=logging_training_loss_every))
    def log_training_loss(trainer):
        logger.info(f"(Progress) Loss: {trainer.state.output}")

    @trainer.on(Events.EPOCH_COMPLETED)
    @trainer.on(Events.STARTED)
    def log_eval_results():
        logger.info(f"(Progress) Will evaluate model on dev split")
        evaluator.run(loader_validation)
        m = evaluator.state.metrics[name_metric]
        logger.info(f"(Progress) Eval Results - Epoch: {trainer.state.epoch}  {name_metric}: {m}")
        logger.info(f"(Progress) Early Stopping State (Counter / Patience): "
                    f"{handler_early_stopping.counter}/{handler_early_stopping.patience}")

    # start train job
    trainer.run(loader_train, max_epochs=max_epochs)
    logger.info(
        f'(Progress) Terminated train after {trainer.state.epoch} w/ {name_metric} {handler_early_stopping.best_score}')
    logger.info(f'(Progress) Saved best model to {handler_model_checkpoint.last_checkpoint}')

    return handler_model_checkpoint.last_checkpoint
