# the base directory
local path_base_dir = "$HOME/experiments/gxai/2021-03-08-experiment-imdb";
# whether or not to normalize attribution scores on sequence level
local normalize = true;
# the name of the encoder
local name_model = "bert-base-cased";
# for the heading of each explanation
local name_model_html = "BERT (base, cased)";
# also for the heading of each explanation
local name_dataset_html = "IMDB (Test Split)";

# the first set of explanations
local datasets = [
{
    # if loaded from json lines, use local.explanations
    "name": "local.explanations",
    "name_explainer": "integrated-gradients",
    # the number of samples used, is used internally
    "n_samples": 20,
    # index to label mapping
    "labels": {"0": "negative", "1": "positive"},
    "config": {
        "paths_json_files": path_base_dir + '/explanations/' +
        "bert-base-cased.huggingface.imdb.test.integrated-gradients.samples-"+ $['n_samples'] + ".batch-1to-1.jsonl",
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
        "bert-base-cased.huggingface.imdb.test.empirical-explainer.ig-20.batch-1to-1.jsonl",
        "batch_size": 1, # do not change
        "columns": ["attributions", "input_ids"],
        },
    },
];


{
    "path_file_out": path_base_dir + '/visualizations/' + "heatmaps.imdb.test" + '.html',
    "datasets": datasets,
    "name_model": name_model,
    "name_model_html": name_model_html,
    "name_dataset_html": name_dataset_html,
    "normalize": normalize,
    "position_pad": 'back',
}

