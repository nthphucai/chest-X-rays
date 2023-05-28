#!/bin/bash
set -e

python classifier/data/prepare_data.py \
    --data_path data/vinbigdata/vin_train_val.csv \
    --out_path /content/drive/MyDrive/Classification2D/chest-X-rays/data \
    --config_path configs/preprocess_pipeline.yaml \
    --classes data/vinbigdata/vin_classes_14.npy \
    --train_file_name train_dataset.pt \
    --valid_file_name valid_dataset.pt \
    --fold 1 \
