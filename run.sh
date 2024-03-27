#!/bin/bash

python main.py \
    --trainer configs/trainer/default.yaml \
    --model configs/models/mlp.yaml \
    --data configs/data_modules/har.yaml