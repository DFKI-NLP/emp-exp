#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

pip3 install -r requirements.txt

python run_job.py -c ./configs/paws/download/download.paws.train-pt1.jsonnet
python run_job.py -c ./configs/paws/download/download.paws.train-pt2.jsonnet
python run_job.py -c ./configs/paws/download/download.paws.train-pt3.jsonnet
python run_job.py -c ./configs/paws/download/download.paws.train-pt4.jsonnet
python run_job.py -c ./configs/paws/download/download.paws.train-pt5.jsonnet
python run_job.py -c ./configs/paws/download/download.paws.train-pt6.jsonnet
