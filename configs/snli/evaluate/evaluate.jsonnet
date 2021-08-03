local BATCH_SIZE = 1; // Do not change

local path_base_dir = "/netscratch/feldhus/experiments/emp-exp/snli/";
local name_file_target = "xlnet-base-cased.huggingface.snli.test.shapley-sampling.samples-20.batch-1to-1.jsonl";
local filenames_ = [path_base_dir + "explanations/"  + "xlnet-base-cased.huggingface.snli.test.shapley-sampling.samples-"+samples +".batch-1to-1.jsonl" for samples in std.range(1,20)];
local filenames = filenames_ + [path_base_dir + "explanations/" + "xlnet-base-cased.huggingface.snli.test.empirical-explainer.svs-20.batch-1to-1.jsonl"];

# derived automatically
local dataset_target = {
    "name": "local.explanations",
    "config": {
        "paths_json_files": path_base_dir + "explanations/" + name_file_target,
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