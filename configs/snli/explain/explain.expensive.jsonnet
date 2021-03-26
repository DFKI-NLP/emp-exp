local base_directory = "$HOME/experiments/gxai/2021-02-26-experiment-snli";

local name_model = 'xlnet-base-cased';
local name_file_model = "2021-02-26-15-37-03.xlnet-base-cased.huggingface.snli_model_f1=0.9124.pt";

local name_dataset = 'huggingface.snli';
local name_split = "test";
local start = -1;
local end = -1;
local batch_size = 1; # must be 1 due to implementation details and memory issues

local name_explainer =  "shapley-sampling";
local internal_batch_size = 1; # must be 1 due to implementation details and memory issues, later set by Captum
local num_labels = 3;
local num_samples = 10;
local early_stopping = -1;
local max_length = 130;


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
      "columns": ['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'special_tokens_mask'],
      "batch_size": batch_size,
    },
};

local explainer = {
    "config": {
      "internal_batch_size": internal_batch_size,
      "mode_load": "ignite",
      "n_samples": num_samples,
      "name_model": name_model,
      "path_model": path_model, # path_dir_base + name_file_model,
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
