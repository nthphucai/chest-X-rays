#!/bin/bash
set -e

python classifier/data/prepare_data.py \
    --data_path /content/drive/MyDrive/Classification2D/NIH-Chest-X-ray-Dataset/train_val_data_25000.csv \
    --out_path /content/drive/MyDrive/Classification2D/chest-X-rays/data \
    --config_path configs/preprocess_pipeline.yaml \
    --train_file_name train_dataset.pt \
    --valid_file_name valid_dataset.pt \
    --fold 1 \
    --classes /content/drive/MyDrive/Classification2D/chest-X-rays/data/nih/nih_classes_14.npy 