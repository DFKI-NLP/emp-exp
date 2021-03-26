# base directory w/o trailing backslash
local base_directory = "$HOME/experiments/gxai/2021-02-26-experiment-snli";

# dataset
local name_dataset = 'huggingface.snli';
local name_model = 'xlnet-base-cased';
local name_split = "test";
local start = -1;
local end = -1;
local batch_size = 16;

# explainer
local name_explainer =  "empirical-explainer"; # "shapley-sampling"; #"integrated-gradients";
local name_target_explainer = "svs-20";
local max_length = 130;
local name_file_model = "2021-03-17-13-55-30.empirical-explainer.local.explanations_model_nmse=-1.8915.pt";
local dim_embedding = 768;
local attend_on_pad_tokens = true;
local name_encoder = "xlnet-base-cased";

# automatically derived:
local path_dataset = base_directory + "/data/"+name_split+"/";
local path_out = base_directory + "/explanations/"+name_model+"."+name_dataset+"."+name_split+"."+ name_explainer +
"." + name_target_explainer + "." + "batch" + start + "to" + end + ".jsonl";


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
      "dim_embedding": dim_embedding,
      "downstream": null,
      "input_keys": [
        "input_ids",
        "attention_mask",
        "token_type_ids"
      ],
      "name_encoder": name_encoder,
      "attend_on_pad_tokens": attend_on_pad_tokens,
      # "name_model": "xlnet-base-cased",
      "path_model": base_directory + "/models/" + name_file_model,
      "seq_len": max_length,
    },
    "name": "empirical-explainer"
  };

{
    'explainer': explainer,
    'dataset': dataset,
    'start': start,
    'end': end,
    'batch_size': batch_size,
    'path_out': path_out,
}
