{
    // name of the dataset according to HF datasets
    "name": "imdb",
    // the split to download; since there is no validation split for IMDB we handle the split ourselves
    "split": "train[11250:12500]+train[23750:]",
    // the directory where to save the dataset, $HOME will be replaced by your home directory
    "path_out": "$HOME/experiments/gxai/2021-02-26-experiment-imdb/data/validation/",
}
