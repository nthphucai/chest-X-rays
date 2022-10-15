#!/bin/bash
set -e

python run.py \
    --model_path output/checkpoints/fold0_20221002.pt \
    --data_path output/csv/data_train.csv \
    --config_path configs/config.json \
    --output_dir output \
    --do_train True \
    --do_eval False \
    --train_batch_size 3 \
    --eval_batch_size 2 \
    --save_model True 
