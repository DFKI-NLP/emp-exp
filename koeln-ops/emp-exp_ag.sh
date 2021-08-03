#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

pip3 install -r requirements.txt

python run_job.py -c ./configs/ag_news/train/train.empirical-explainer.jsonnet 2>&1 | tee -a ./logs/emp-exp_ag.log
