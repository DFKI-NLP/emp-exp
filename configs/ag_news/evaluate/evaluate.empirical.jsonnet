local BATCH_SIZE = 1; // Do not change
# the base dir
local path_base_dir = "/netscratch/feldhus/experiments/emp-exp/ag_news/";
# name of the target explanations, assumed in <base-dir>/explanations/
local name_file_target = "roberta-base.huggingface.ag_news.test.integrated-gradients.samples-25.batch-1to-1.jsonl";
# the approximative explanations
local filenames_ = [path_base_dir + "explanations/"  + "roberta-base.huggingface.ag_news.test.integrated-gradients.samples-"+ samples +".batch-1to-1.jsonl" for samples in std.range(1,25)];
#local filenames = filenames_;  # comment out this line and uncomment the next as soon as empirical explanations are available
local filenames = filenames_ + [path_base_dir + "explanations/" + "roberta-base.huggingface.ag_news.test.empirical-explainer.ig-25.batch-1to-1.jsonl"];

local dataset_target = {
    "name": "local.explanations",
    "config": {
        "paths_json_files": path_base_dir + "/explanations/" + name_file_target,
        "columns": ["input_ids", "attributions"],
        "batch_size": BATCH_SIZE,
    },
};

{
    "dataset_target": dataset_target,
    "filenames": filenames,
    "compute_convergence": true,
    "compute_f1": true,
}