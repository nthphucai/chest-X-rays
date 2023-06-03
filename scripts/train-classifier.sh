#!/bin/bash
set -e

python classifier/training/train.py \
    --model_name_or_path chexnet \
    --model_type chexnet \
    --freeze_feature False \
    --output_dir output/models/fold_1/best_model_0106.pt \
    --config_dir configs/classifier_config.yaml \
    --use_gcn True \
    --train_dataset_path output/dataset/train_dataset.pt \
    --valid_dataset_path output/dataset/valid_dataset.pt \
    --num_train_epochs 5 \
    --log_dir output/logs_2403.csv \
    --do_train True \
    --do_eval False \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --report_to wandb \

