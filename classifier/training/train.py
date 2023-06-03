import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch

import wandb
from classifier.models.classify import classifier
from classifier.training.trainer.config_runner import ConfigTrainer
from classifier.utils.file_utils import logger, read_yaml_file
from classifier.utils.hf_argparser import HfArgumentParser


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default="chexnet",
        metadata={"help": "Path to pretrained model or model name"},
    )

    model_type: Optional[str] = field(
        default="chexnet",
        metadata={"help": "Specify the type of model"},
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models"},
    )

    freeze_feature: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze_feature"},
    )

    output_dir: str = field(
        default=None,
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )

    config_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    num_classes: Optional[int] = field(
        default=14, metadata={"help": "Number of classed to be classfied"}
    )

    use_gcn: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply gcn"},
    )


@dataclass
class DataTrainingArguments:
    train_dataset_path: Optional[str] = field(
        default="output/data/train_dataset.pt",
        metadata={"help": "Path to train dataset"},
    )

    valid_dataset_path: Optional[str] = field(
        default="output/data/valid_dataset.pt",
        metadata={"help": "Path to valid dataset"},
    )


@dataclass
class TrainingArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    per_device_train_batch_size: int = field(
        default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    num_train_epochs: int = field(
        default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(
        default=False, metadata={"help": "Whether to run eval on the dev set."}
    )

    save_model: bool = field(default=False, metadata={"help": "Whether to save model."})

    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )

    log_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models"},
    )

    report_to: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "The list of integrations to report the results and logs to."
        },
    )


def runner(
    train_dataset_path: str,
    valid_dataset_path: Optional[str],
    config_dir: str,
    model_name_or_path: str,
    model_type: str = "chextnext",
    cache_dir: Optional[str] = None,
    freeze_feature: bool = False,
    num_classes: int = 14,
    use_gcn: bool = False,
    num_train_epochs: str = 2,
    out_dir: str = None,
    log_dir: str = None,
    fp16: bool = False,
    do_train: bool = True,
    do_eval: bool = False,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    report_to: str = "wandb",
):
    if not os.path.exists(os.path.dirname(log_dir)):
        os.makedirs(os.path.dirname(log_dir))

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logging.warning(
        "Device: %s, n_gpu: %s",
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        torch.cuda.device_count(),
    )
    # Setup wandb
    if report_to is not None:
        if report_to == "wandb":
            wandb.login()
            wandb.init(
                project="kidney-segment",
                name=model_name_or_path,
                group=model_type,
                tags=["baseline", "unet"],
                job_type="train",
            )

    config = read_yaml_file(config_dir)["classifier"]

    model = classifier(
        backbone=model_name_or_path,
        gcn=use_gcn,
        pretrained_path=cache_dir,
        freeze_feature=freeze_feature,
        n_class=num_classes,
    )

    train_dataset = torch.load(train_dataset_path)
    valid_dataset = torch.load(valid_dataset_path)

    logger.info("The number of train samples: %s", len(train_dataset))
    logger.info("The number of eval samples: %s", len(valid_dataset))

    trainer = ConfigTrainer(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        model=model,
        config=config,
        save_config_path=None,
        verbose=False,
        num_train_epochs=num_train_epochs,
        out_dir=out_dir,
        log_dir=log_dir,
        fp16=fp16,
        do_train=do_train,
        do_eval=do_eval,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
    )
    trainer.train()

    logger.info("Save logs at directory: %s", log_dir)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    runner(
        train_dataset_path=data_args.train_dataset_path,
        valid_dataset_path=data_args.valid_dataset_path,
        config_dir=model_args.config_dir,
        model_name_or_path=model_args.model_name_or_path,
        model_type=model_args.model_type,
        out_dir=model_args.output_dir,
        cache_dir=model_args.cache_dir,
        freeze_feature=model_args.freeze_feature,
        num_classes=model_args.num_classes,
        use_gcn=model_args.use_gcn,
        num_train_epochs=training_args.num_train_epochs,
        log_dir=training_args.log_dir,
        fp16=training_args.fp16,
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        report_to=training_args.report_to,
    )


if __name__ == "__main__":
    main()
