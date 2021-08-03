#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

pip3 install -r requirements.txt

python run_job.py -c ./configs/paws/evaluate/evaluate.empirical.jsonnet 2>&1 | tee -a ./logs/ev-emp_paws.log
