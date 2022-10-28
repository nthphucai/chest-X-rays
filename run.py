import os
import sys
import pandas as pd 
import torch 
from typing import Optional, List
import wandb
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass, field
from reposcv.training.trainer.standard_trainer import Trainer
from reposcv.training.data.get_dl import get_dloader
from reposcv.training.trainer.config_runner import ConfigTrainer
from classifer.classifier import classifier

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

import logging
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from ..."},
    )
    
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models"},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )

@dataclass
class TrainingArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    output_dir: str = field(
        default = None, 
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    report_to: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    )
    train_batch_size: int = field(
        default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    eval_batch_size: int = field(
        default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    config_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )

    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    
    save_model: bool = field(default=False, metadata={"help": "Whether to save model."})

    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )

def main(args_file=None):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if (len(sys.argv) == 2 and sys.argv[1].endswith(".json")) or args_file is not None:
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args_file_path = os.path.abspath(sys.argv[1]) if args_file is None else args_file
        model_args, data_args, training_args = parser.parse_json_file(json_file=args_file_path)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO #if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    logger.warning(
        "Device: %s, n_gpu: %s",
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        torch.cuda.device_count(),
    )

    # Set seed
    set_seed(training_args.seed)
    
    # Setup wandb
    if training_args.report_to is not None:
        if training_args.report_to == "wandb":
            wandb.login()
            wandb.init(
                project="chexnet-classifier",
                name=model_args.model_name_or_path,
                group=model_args.model_type,
                tags=["baseline", "chextnet"],
                job_type="train",
            )

    # Load datasets
    logger.info("Loading dataset")
    data = pd.read_csv(data_args.data_path, index_col=0)
    train_dataloader, val_dataloader  = get_dloader(df=data, fold=0)

    # Load pretrained model
    model = classifier(gcn = False, pretrained_path = model_args.model_path, freeze_feature = False, n_class=14)

    # Initialize our Trainer
    configs = ConfigTrainer(model_config = model, config_path=training_args.config_path, verbose=None)()
    trainer = Trainer(
                model = model, 
                train_data = train_dataloader,
                val_data = val_dataloader,  
                loss = configs["loss"], 
                optimizer = configs["opt"], 
                scheduler = configs["scheduler"], 
                metric = configs["metric"],
                num_train_epochs = training_args.train_batch_size,
                output_dir = training_args.output_dir,
                save_model = training_args.save_model,
                fp16 = training_args.fp16
    )
    logging.getLogger("wandb.run_manager").setLevel(logging.WARNING)

    # Training
    if training_args.do_train:
        trainer.run(mode = ["train"])

    # Evaluation 
    if training_args.do_eval:
        logger.info("*** Evaluation ***")
        trainer.run(mode = ["valid"])

if __name__ == "__main__":
    main()

