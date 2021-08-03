# base directory w/o trailing backslash
local base_directory = "/netscratch/feldhus/experiments/emp-exp/ag_news";

# name of the encoder used in the downstream model, other option xlnet-base-cased
local name_model = 'roberta-base';
# file name of the model weights, expected in <base-dir>/models/
local name_file_model = "2021-07-26-13-11-23.roberta-base.huggingface.ag_news_model_f1=0.9402.pt";

# identifier of the dataset, should comply to the training data of the downstream model, see train.downstream.jsonnet
local name_dataset = 'huggingface.ag_news';
# split of the trainign data to explain, expected in <base-dir>/data/name_split/
local name_split = "validation";
# start index of the batch to explain, -1 for all
local start = -1;
# end index of the batch to explain, -1 for all
local end = -1;
# the batch size to use for explaining
local batch_size = 1; # must be 1 due to implementation details and memory issues

# name of the expensive explainer, other option: shapley-sampling
local name_explainer =  "integrated-gradients";
# the number of output neurons of the downstream model
local num_labels = 4;
# number of samples
local num_samples = 25;
# early stopping index, can be used to interrupt explanations, -1 for all
local early_stopping = -1;
# maximum input sequence length to the downstream model
local max_length = 512;
# internal batch size, fixed at one to solve memory issues
local internal_batch_size = 1; # must be 1 due to implementation details and memory issues, later set by Captum

# automatically derived:
local path_dataset = base_directory + "/data/"+name_split+"/";
local path_model = base_directory + "/models/" + name_file_model;
local path_out = base_directory + "/explanations/"+name_model+"."+name_dataset+"."+name_split+"."+ name_explainer +
".samples-"+ num_samples +"." + "batch" + start + "to" + end + ".jsonl";


local dataset = {
    "name": name_dataset,
    "config": {
      "name_model": name_model,
      "path_dataset": path_dataset,
      "start": start,
      "end": end,
      "max_length": max_length,
      "columns": ['input_ids', 'attention_mask', 'labels', 'special_tokens_mask'],
      "batch_size": batch_size,
    },
};

local explainer = {
    "config": {
      "internal_batch_size": internal_batch_size,
      "mode_load": "ignite",
      "n_samples": num_samples,
      "name_model": name_model,
      "path_model": path_model,
      "num_labels": num_labels,
    },
    "name": name_explainer,
  };


{
    'explainer': explainer,
    'dataset': dataset,
    'start': start,
    'end': end,
    'batch_size': batch_size,
    'early_stopping': early_stopping,
    'path_out': path_out,
}
