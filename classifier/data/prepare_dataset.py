import os
import pathlib
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch

from classifier.data.augs import aug_maps
from classifier.data.data_readers.data_reader import ChextNextDataset
from classifier.data.utils import split_data
from classifier.utils.file_utils import logger
from classifier.utils.hf_argparser import HfArgumentParser


@dataclass
class DataTrainingArguments:
    data_path: Optional[str] = field(
        default="data/train_ds.csv", metadata={"help": "Path for data directory"}
    )

    out_path: Optional[str] = field(
        default=None, metadata={"help": "Path for output directory"}
    )

    train_file_name: Optional[str] = field(
        default="train_dataset.pt", metadata={"help": "Name for cached train dataset"}
    )

    valid_file_name: Optional[str] = field(
        default="valid_dataset.pt", metadata={"help": "Name for cached valid dataset"}
    )

    test_file_name: Optional[str] = field(
        default="test_dataset.pt", metadata={"help": "Name for cached test dataset"}
    )

    fold: Optional[int] = field(default=None, metadata={"help": "fold"})

    classes: Optional[str] = field(
        default="output/columns_14.npy",
        metadata={"help": "Path for classes file directory"},
    )


def get_dataset(
    data_path: str,
    fold: int = 1,
    out_path: str = None,
    train_file_name: str = "train_dataset.pt",
    valid_file_name: str = "valid_dataset.pt",
    classes: str = "output/columns_14.npy",
):
    random.seed(42)

    if ".csv" not in pathlib.Path(data_path).suffix:
        raise "Only support .csv extension"

    df = pd.read_csv(data_path)[:1000]
    df = split_data(df, n_split=10)
    logger.info(f"The number of data at {df.shape[0]}")

    if fold is not None:
        train_df = df.loc[df["fold"] != fold].reset_index(drop=True)
        valid_df = df.loc[df["fold"] == fold].reset_index(drop=True)
    else:
        train_df = df.reset_index(drop=True)
        # default fold=0 for valid data
        valid_df = df.loc[df["fold"] == 0].reset_index(drop=True)

    columns = np.load(classes, allow_pickle=True)
    train_ds = ChextNextDataset(df=train_df, augs=aug_maps["train_augs"], columns=columns)
    valid_ds = ChextNextDataset(df=valid_df, augs=aug_maps["val_augs"], columns=columns)

    train_file_path = os.path.join(out_path, train_file_name)
    torch.save(train_ds, train_file_path)
    logger.info(f"The number of train dataset is {train_df.shape[0]}")
    logger.info(f"Saved train dataset at {train_file_path}")

    valid_file_path = os.path.join(out_path, valid_file_name)
    torch.save(valid_ds, valid_file_path)
    logger.info(f"The number of valid dataset is {valid_df.shape[0]}")
    logger.info(f"Saved valid dataset at {valid_file_path}")
    return train_ds, valid_ds


def main():
    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()[0]

    get_dataset(
        data_path=data_args.data_path,
        fold=data_args.fold,
        out_path=data_args.out_path,
        train_file_name=data_args.train_file_name,
        valid_file_name=data_args.valid_file_name,
        classes=data_args.classes,
    )


if __name__ == "__main__":
    main()
