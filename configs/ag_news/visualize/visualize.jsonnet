# the base directory
local path_base_dir = "/netscratch/feldhus/experiments/emp-exp/ag_news";
# whether or not to normalize attribution scores on sequence level
local normalize = true;
# the name of the encoder
local name_model = "roberta-base";
# for the heading of each explanation
local name_model_html = "RoBERTa (base)";
# also for the heading of each explanation
local name_dataset_html = "AG News (Test Split)";

# the first set of explanations
local datasets = [
{
    # if loaded from json lines, use local.explanations
    "name": "local.explanations",
    "name_explainer": "integrated-gradients",
    # the number of samples used, is used internally
    "n_samples": 25,
    # index to label mapping
    # World (0), Sports (1), Business (2), Sci/Tech (3).
    "labels": {"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"},
    "config": {
        "paths_json_files": path_base_dir + '/explanations/' +
        "roberta-base.huggingface.ag_news.test.integrated-gradients.samples-"+ $['n_samples'] + ".batch-1to-1.jsonl",
        "batch_size": 1, # do not change
        "columns": ["attributions", "predictions", "input_ids", "labels"],
    },
},
{
    # if loaded from json lines, use local.explanations
    "name": "local.explanations",
    "name_explainer": "integrated-gradients",
    # the number of samples used, is used internally
    "n_samples": 6,
    # index to label mapping
    # World (0), Sports (1), Business (2), Sci/Tech (3).
    "labels": {"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"},
    "config": {
        "paths_json_files": path_base_dir + '/explanations/' +
        "roberta-base.huggingface.ag_news.test.integrated-gradients.samples-"+ $['n_samples'] + ".batch-1to-1.jsonl",
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
        "roberta-base.huggingface.ag_news.test.empirical-explainer.ig-25.batch-1to-1.jsonl",
        "batch_size": 1, # do not change
        "columns": ["attributions", "input_ids"],
        },
    },
];


{
    "path_file_out": path_base_dir + '/visualizations/' + "heatmaps.ag_news.test" + '.html',
    "datasets": datasets,
    "name_model": name_model,
    "name_model_html": name_model_html,
    "name_dataset_html": name_dataset_html,
    "normalize": normalize,
    "position_pad": 'back',
}

