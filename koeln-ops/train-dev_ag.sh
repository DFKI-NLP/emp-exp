#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

pip3 install -r requirements.txt

python run_job.py -c ./configs/ag_news/explain/explain.expensive.train.jsonnet
python run_job.py -c ./configs/ag_news/explain/explain.expensive.validation.jsonnet