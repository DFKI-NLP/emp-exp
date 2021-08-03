# the base directory
local base_directory = "/netscratch/feldhus/experiments/emp-exp/paws";

# dataset
# which dataset to explain
local name_dataset = 'huggingface.paws';
# which split to explain
local name_split = "test";

# start of the batch to explain, -1 for all
local start = -1;
# end of the batch to explain, -1 for all
local end = -1;
# the batch size to use during explanations
local batch_size = 4;

# name of the explainer
local name_explainer =  "empirical-explainer";
# abbreviation for the target explainer, used for file naming purposes
local name_target_explainer = "svs-20";
# the name of the encoder used for the downstream model and the empirical explainer, other option: xlnet-base-cased
local name_encoder = "google/electra-small-discriminator";
# the maximum input sequence length to the downstream model
local max_length = 145;
# the file name of the empirical explainer (weights), assumed in <base-dir>/models/
local name_file_model = "2021-08-03-11-22-43.empirical-explainer.local.explanations_model_nmse=-0.6566.pt";
# encoder output dimension
local dim_embedding = 256;
# whether or not to attend on padding tokens
local attend_on_pad_tokens = true;
# number of labels
local num_labels = 2;

local name_encoder_str = std.strReplace(name_encoder, '/', '-');

# automatically derived:
local path_dataset = base_directory + "/data/"+name_split+"/";
local path_out = base_directory + "/explanations/"+name_encoder_str+"."+name_dataset+"."+name_split+"."+ name_explainer +
"." + name_target_explainer + "." + "batch" + start + "to" + end + ".jsonl";


local dataset = {
    "name": name_dataset,
    "config": {
      "name_model": name_encoder,
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
        "attention_mask", // todo: review!
        "token_type_ids",
      ],
      "attend_on_pad_tokens": attend_on_pad_tokens,
      "name_encoder": name_encoder,
      "num_labels": num_labels,
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
