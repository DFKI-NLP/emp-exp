// define only once
local name_model = 'xlnet-base-cased';
local name_dataset = 'huggingface.snli';
local max_length = 130;
local num_labels = 3;
local batch_size_train = 16;
local batch_size_eval = 32;
local patience = 2;
local max_epochs = 100;
local path_dir_model = '$HOME/experiments/gxai/2021-02-26-experiment-snli/models/';
local path_dataset_train = "$HOME/experiments/gxai/2021-02-26-experiment-snli/data/train/";
local path_dataset_validation = "$HOME/experiments/gxai/2021-02-26-experiment-snli/data/validation/";
local logging_training_loss_every = 10;

local dataset_huggingface_train = {
    "name": name_dataset,
    "config": {
      "name_model": name_model,
      "path_dataset": path_dataset_train,
      "max_length": max_length,
      "columns": ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
      "batch_size": batch_size_train,
    },
  };

local dataset_huggingface_validation = {
    "name": name_dataset,
    "config": {
      "name_model": name_model,
      "path_dataset": path_dataset_validation,
      "max_length": max_length,
      "columns": ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
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
