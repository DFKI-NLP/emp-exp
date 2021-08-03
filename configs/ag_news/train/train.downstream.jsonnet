// the name of the encoder, must be registered w/ HF transformers
local name_model = 'roberta-base'; // this code base was developed for and tested with only bert-base-cased and xlnet-base-cased
// a dataset identifier, current options: huggingface.imdb, huggingface.snli, local.explanations (for json lines)
local name_dataset = 'huggingface.ag_news';
// the maximum sequence length
local max_length = 512;
// the number of output neurons
local num_labels = 4;
// batch size during training, decrease if out-of-memory errors occur
local batch_size_train = 8;
// batch size during evaluation, decrease if out-of-memory errors occur
local batch_size_eval = 16;
// patience for early stopping
local patience = 2;
// the maximum number of epochs after which to terminate training
local max_epochs = 100;
// the directory to where to save the model parameters
local path_dir_model = '/netscratch/feldhus/experiments/emp-exp/ag_news/models/';
// the directory from where to load the training data
local path_dataset_train = "/netscratch/feldhus/experiments/emp-exp/ag_news/data/train/";
// the directory from where to load the validation data
local path_dataset_validation = "/netscratch/feldhus/experiments/emp-exp/ag_news/data/validation/";
// the interval at which to log information, such as loss
local logging_training_loss_every = 10;


// derived automatically, do not change
local dataset_huggingface_train = {
    "name": name_dataset,
    "config": {
      "name_model": name_model,
      "path_dataset": path_dataset_train,
      "max_length": max_length,
      "columns": ['input_ids', 'attention_mask', 'labels'],
      "batch_size": batch_size_train,
    },
  };

local dataset_huggingface_validation = {
    "name": name_dataset,
    "config": {
      "name_model": name_model,
      "path_dataset": path_dataset_validation,
      "max_length": max_length,
      "columns": ['input_ids', 'attention_mask', 'labels'],
      "batch_size": batch_size_eval,
    },
  };

local model = {
    "name": name_model,
    "config": {
        "num_labels": num_labels,
    },
};

{
    "model": model,
    "dataset_train": dataset_huggingface_train,
    "dataset_validation": dataset_huggingface_validation,
    "batch_size_train": batch_size_train,
    "batch_size_validation": batch_size_eval,
    "max_epochs": max_epochs,
    "patience": patience,
    "path_dir_model": path_dir_model,
    "logging_training_loss_every": logging_training_loss_every,
}
