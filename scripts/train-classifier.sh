#!/bin/bash
set -e

python classifier/training/train.py \
    --model_name_or_path chexnet \
    --train_dataset_path output/dataset/train_dataset.pt \
    --valid_dataset_path output/dataset/valid_dataset.pt \
    --config_dir configs/classifier_config.yaml \
    --output_dir output/models/fold_1/best_model_0106.pt \
    --num_train_epochs 5 \
    --log_dir output/logs_2403.csv \
    --freeze_feature False \
    --do_train True \
    --do_eval False \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --report_to wandb \

