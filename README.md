# Empirical Explainers 
This repo contains the refactored code and pointers to the data accompanying our paper ["Efficient Explanations from Empirical Explainers"](https://arxiv.org/abs/2103.15429). 

Below, please find (1) pointers to the heatmaps promised in the paper, (2) instructions on how to replicate our experiments and (3) all our data, models and logs. 

Disclaimer: This is a refactored and prettified version of the code and has not been tested exhaustively yet. This disclaimer will be removed when the tests have been conducted.

##  :floppy_disk: Download the Explanations 
Here are the attribution maps by the expensive target explainers and the Empirical Explainers. Each line contains an HTML document. 
Your browser probably will not be able to load all data at once. You can extract individual lines into separate files to solve that issue. 
* [BERT / IMDB / Integrated Gradients](https://cloud.dfki.de/owncloud/index.php/s/2o9cjKCp8tmJHFr/download) (823 MB)
* [XLNet / SNLI / Shapley Value Samples](https://cloud.dfki.de/owncloud/index.php/s/8iiMRLLcNaccLen/download) (37 MB)
* [RoBERTa / AG News / Integrated Gradients](https://cloud.dfki.de/owncloud/index.php/s/RTJZ6pQ4pEXS6b9/download) (52 MB)
* [ELECTRA / PAWS / Shapley Value Samples](https://cloud.dfki.de/owncloud/index.php/s/yLpFDxBmJQGwjz5/download) (53 MB)



## :bar_chart: Replicate our Experimental Results
You can replicate our experimental results, following the steps below or download our data, model and logs, following the
links at the bottom. Our experiments are config-driven. 

### :file_folder: Directories and File Structure 
We assume a base directory for all experiments with 5 sub folders: 

1. |- `data`: The raw data is downloaded here, using [Huggingface's (HF) datasets](https://huggingface.co/datasets). 
2. |- `models`: Model parameters are saved here. 
3. |- `explanations`: Saliency maps are saved here as json lines. 
4. |- `visualizations`: We save attribution maps in html format here. 
5. |- `logs`: Training, explanation and visualization logs are saved here. 

### :books: Install the dependencies
1. Install pytorch / torchvision 
2. Install the requirements listed in `requirements.txt`
   * A complete list of the exact requirements is contained in `pip.freeze.txt`

### :running: Running jobs
All jobs are coordinated using the flags and pointers in `./configs/run_job.jsonnet`. 
To run a job, adapt the config and then run, e.g. 

`CUDA_VISIBLE_DEVICES=0 python run_job.py 2>&1 | tee -a <base-dir>/logs/job.log`

### :floppy_disk: Download the Data 
In `run_job.jsonnet`, please set the `download` flag to true 
and let `path_config_download` point to a download configuration file. Example files are provided in
`./configs/snli/download/` and `./configs/imdb/download/`. We annotated 

`./configs/imdb/download/download.imdb.validation.jsonnet`

We save the training data to `|-data/train/`, the validation data to `|-data/validation/`and the test data to `|data/test/`. 

### :computer: Train the downstream model 
We train models using [PyTorch Ignite](https://pytorch.org/ignite/) in conjunction w/ [HF's Transformers](https://huggingface.co/transformers/). 

The training is again config-driven. We annotated the fields in

`./configs/imdb/train/train.downstream.jsonnet`. 

After training, the model can be found in `|-models`. 

### :factory: Expensively explain the downstream model
We explain models using [PyTorch's Captum](https://github.com/pytorch/captum).
The training is again config driven, an annotated config is provided in 

`./configs/imdb/explain/explain.expensive.jsonnet`

Explanations are written to json lines. The json file can be found in `|-explanations/`, after the job is done. 
At a minimum, an explanation contains the fields `input_ids` and `attributions`.

### :computer: Train the Empirical Explainer
The training of the Empirical Explainer uses the same script as the training of the downstream model.
An annotated config file is provided in 

`./configs/train/train.empirical-explainer.jsonnet`

The model weights can be found in `|-models/` after training. 

### :sunrise_over_mountains: Empirically explain the downstream model
The efficient explanations are generated with the same script as the expensive explanations. An annotated config file is 
provided in 

`./configs/imdb/explain/explain.empirical.jsonnet`

Explanations are written to json lines again, which can be found in `|-explanations` after the job is done. 

### :mortar_board: Evaluate 
Evaluation statistics are logged, which requires you to pipe the logger output into a log, e.g. using 

`2>&1 | tee -a <base-dir>/logs/my-log.log`

An example config file is provided 

`./configs/imdb/evaluate/evaluate.jsonnet`

The MSEs between the specified target explanations and the approximative explanations are logged, 
as is the weighted F1 score, which is derived from the true labels and predictions that should be contained 
in the expensive target explanations. 

### :art: Visualize
Finally, you can visualize explanations. For this, specify one or many explanations to load. It is assumed that 
the instances contained in the json lines appear in the same order in the files you specified. 

We again provide an annotated config file 

`./configs/visualize/visualize.jsonnet`

After the job is done, the sub folder `|-visualizations/` contains the heatmaps. The heatmaps are written to a file, 
line by line, where each line is an HTML document that contains the explanations. 

## Download our Data, Models and Logs
Our experiments (data, models, logs) can be downloaded, here:

* [BERT/IMDB/Integrated Gradients](https://cloud.dfki.de/owncloud/index.php/s/3nJsWyq5mzCjMq3) (10 GB :warning:)
* [XLNet/SNLI/Shapley Values](https://cloud.dfki.de/owncloud/index.php/s/F6RYC2xpT8wmb9i) (2.5 GB :warning:)
* [RoBERTa/AG News/Integrated Gradients](https://cloud.dfki.de/owncloud/index.php/s/XBygJL2mR9gfHJa) (2 GB :warning:)
* [ELECTRA/PAWS/Shapley Values](https://cloud.dfki.de/owncloud/index.php/s/fHX5iwDb8xKZYkx) (1 GB :warning:)
