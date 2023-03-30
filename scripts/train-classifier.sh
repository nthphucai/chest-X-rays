#!/bin/bash
set -e

python classifier/training/run.py \
    --train_dataset_path data/valid_dataset.pt \
    --valid_dataset_path data/valid_dataset.pt \
    --config_dir configs/classifier_config.yaml \
    --output_dir output/checkpoints \
    --num_train_epochs 5 \
    --log_dir output/logs_2403.csv \
    --freeze_feature False \
    --do_train True \
    --do_eval True \
