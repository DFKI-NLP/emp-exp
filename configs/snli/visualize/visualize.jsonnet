local path_base_dir = "$HOME/experiments/gxai/2021-02-26-experiment-snli";
local normalize = true;
local name_model = "xlnet-base-cased";
local name_model_html = "XLNet (base, cased)";
local name_dataset_html = "SNLI (Test Split)";

local datasets = [
{
    "name": "local.explanations",
    "name_explainer": "shapley-sampling",
    "n_samples": 20,
    "labels": {"0": "entailment", "1": "neutral", "2": "contradiction"},
    "config": {
        "paths_json_files": path_base_dir + '/explanations/' +
        "xlnet-base-cased.huggingface.snli.test.shapley-sampling.samples-"+ $['n_samples'] + ".batch-1to-1.jsonl",
        "batch_size": 1,
        "columns": ["attributions", "predictions", "input_ids", "labels"],
        },
    },
{
    "name": "local.explanations",
    "name_explainer": "empirical",
    "config": {
        "paths_json_files": path_base_dir + '/explanations/' +
        "xlnet-base-cased.huggingface.snli.test.empirical-explainer.svs-20.batch-1to-1.jsonl",
        "batch_size": 1,
        "columns": ["attributions", "input_ids"],
        },
    },
];


{
    "path_file_out": path_base_dir + '/visualizations/' + "heatmaps.snli.test" + '.html',
    "datasets": datasets,
    "name_model": name_model,
    "name_model_html": name_model_html,
    "name_dataset_html": name_dataset_html,
    "normalize": normalize,
    "position_pad": 'front',
}

