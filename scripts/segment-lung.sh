#!/bin/bash
set -e

python classifier/modules/segment_lungs.py \
    --model_name_or_path resnet34 \
    --data_path data/vinbigdata/vin_train_val.csv \
    --output_path output/vin-lung-seg \
    --img_size 256 \
    --batch_size 8
