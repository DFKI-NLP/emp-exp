local path_base_dir = "/netscratch/feldhus/experiments/emp-exp/snli";

# model
local name_model = "xlnet-base-cased";
local name_file_model = "2021-02-26-15-37-03.xlnet-base-cased.huggingface.snli_model_f1=0.9124.pt";
local max_length = 130;
local dim_embedding = 768;
local num_labels = 3;
local attend_on_pad_tokens = true;
local mode_load = "ignite";
# automatically derived:
local path_dir_model = path_base_dir + '/models/';

# dataset
local name_dataset = "local.explanations";
local name_file_validation = "xlnet-base-cased.huggingface.snli.validation.shapley-sampling.samples-20.batch-1to-1.jsonl";

local train_files = [
path_base_dir + "/explanations/" + "xlnet-base-cased.huggingface.snli.train.shapley-sampling.samples-20.batch0to20000.jsonl",
path_base_dir + "/explanations/" + "xlnet-base-cased.huggingface.snli.train.shapley-sampling.samples-20.batch20000to40000.jsonl",
path_base_dir + "/explanations/" + "xlnet-base-cased.huggingface.snli.train.shapley-sampling.samples-20.batch40000to60000.jsonl",
path_base_dir + "/explanations/" + "xlnet-base-cased.huggingface.snli.train.shapley-sampling.samples-20.batch60000to80000.jsonl",
path_base_dir + "/explanations/" + "xlnet-base-cased.huggingface.snli.train.shapley-sampling.samples-20.batch80000to100000.jsonl",
];

# training
local logging_training_loss_every = 10;
local patience = 1;
local max_epochs = 100;
local batch_size_train = 8;
local batch_size_eval = 16;


local dataset_local_explanations_train = {
    "name": "local.explanations",
    "config": {
        "paths_json_files": train_files,
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