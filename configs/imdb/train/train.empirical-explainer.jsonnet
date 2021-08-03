# the base directory
local path_base_dir = "/netscratch/feldhus/experiments/emp-exp/imdb";

# model
# name of the downstream model's encoder, options bert-base-cased, xlnet-base-cased
local name_model = "bert-base-cased";
# file to the downstream model's weights
local name_file_model = "2021-03-08-16-13-44.bert-base-cased.huggingface.imdb_model_f1=0.9324.pt";
# maximum input sequence length to the downstream model
local max_length = 512;
# word embedding dimensions of the downstream models encoder output
local dim_embedding = 768;
# number of output neurons of the downstream model
local num_labels = 2;
# whether or not the empirical explainer shall attend on padding tokens
local attend_on_pad_tokens = true;
# the mode to load the downstream model, ignite or huggingface
local mode_load = "ignite";


# identifier for the dataset, for local json files it is local.explanations
local name_dataset = "local.explanations";
# name of the training set of explanations, assumed in <base-dir>/explanations/
local name_file_train = "bert-base-cased.huggingface.imdb.train.integrated-gradients.samples-20.batch-1to-1.jsonl";
# name of the validation set of explanations, assumed in <base-dir>/explanations/
local name_file_validation = "bert-base-cased.huggingface.imdb.validation.integrated-gradients.samples-20.batch-1to-1.jsonl";


# training
# interval at which to log information
local logging_training_loss_every = 10;
# patience for early stopping
local patience = 1;
# maximum number epochs after which to terminate
local max_epochs = 100;
# the training batch size
local batch_size_train = 8;
# the validation batch size
local batch_size_eval = 8;

# automatically derived:
local path_dir_model = path_base_dir + '/models/';

local dataset_local_explanations_train = {
    "name": "local.explanations",
    "config": {
        "paths_json_files": path_base_dir + "/explanations/" + name_file_train,
        "columns": ["input_ids", "attributions"],
        "batch_size": batch_size_eval,
    },
};

local dataset_local_explanations_validation = {
    "name": "local.explanations",
    "config": {
        "paths_json_files": path_base_dir + "/explanations/" + name_file_validation,
        "columns": ["input_ids", "attributions"],
        "batch_size": batch_size_eval,
    },
};

local model_empirical_explainer = {
    "name": "empirical-explainer",
    "config": {
      "dim_embedding": dim_embedding,
      "downstream": path_base_dir + "/models/" + name_file_model,
      "mode_load": mode_load,
      "name_encoder": name_model,
      "seq_len": max_length,
      "num_labels": num_labels,
      "attend_on_pad_tokens": attend_on_pad_tokens,
    }
};

{
    "model": model_empirical_explainer,
    "dataset_train": dataset_local_explanations_train,
    "dataset_validation": dataset_local_explanations_validation,
    "batch_size_train": batch_size_train,
    "batch_size_validation": batch_size_eval,
    "max_epochs": max_epochs,
    "patience": patience,
    "path_dir_model": path_dir_model,
    "logging_training_loss_every": logging_training_loss_every,
}