# the base directory
local path_base_dir = "/netscratch/feldhus/experiments/emp-exp/paws";
# whether or not to normalize attribution scores on sequence level
local normalize = true;
# the name of the encoder
local name_model = "google/electra-small-discriminator";
# for the heading of each explanation
local name_model_html = "ELECTRA (small)";
# also for the heading of each explanation
local name_dataset_html = "PAWS (Test Split)";

# the first set of explanations
local datasets = [
{
    # if loaded from json lines, use local.explanations
    "name": "local.explanations",
    "name_explainer": "shapley-sampling",
    # the number of samples used, is used internally
    "n_samples": 20,
    # index to label mapping
    "labels": {"0": "no_paraphrase", "1": "paraphrase"},
    "config": {
        "paths_json_files": path_base_dir + '/explanations/' +
        "google-electra-small-discriminator.huggingface.paws.test.shapley-sampling.samples-"+ $['n_samples'] + ".batch-1to-1.jsonl",
        "batch_size": 1, # do not change
        "columns": ["attributions", "predictions", "input_ids", "labels"],
    },
},
{
    # this dictionary can be used to load empirical explanations, which require a different set of fields
    "name": "local.explanations",
    "name_explainer": "empirical", # todo: is not consistent, should be called empirical-explainer, see explain.py
    "config": {
        # path to the file containing the empirical explanations
        "paths_json_files": path_base_dir + '/explanations/' +
        "google-electra-small-discriminator.huggingface.paws.test.empirical-explainer.svs-20.batch-1to-1.jsonl",
        "batch_size": 1, # do not change
        "columns": ["attributions", "input_ids"],
        },
    },
];


{
    "path_file_out": path_base_dir + '/visualizations/' + "heatmaps.paws.test" + '.html',
    "datasets": datasets,
    "name_model": name_model,
    "name_model_html": name_model_html,
    "name_dataset_html": name_dataset_html,
    "normalize": normalize,
    "position_pad": 'back',
}

