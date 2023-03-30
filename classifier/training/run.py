import os
import sys
import pandas as pd 
import torch 
from typing import Optional, List
import wandb
import warnings

from dataclasses import dataclass, field
from classifier.utils.file_utils import logger, read_yaml_file
from classifier.training.trainer.standard_trainer import Trainer
from classifier.training.trainer.config_runner import ConfigTrainer
# from classifier.data.data_loaders import get_dloader
from classifier.models.classify import classifier
from classifier.utils.hf_argparser import HfArgumentParser


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from ..."},
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
        default = None, 
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    config_dir: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )

@dataclass
class DataTrainingArguments:
    train_dataset_path: Optional[str] = field(
        default="output/data/train_dataset.pt", metadata={"help": "Path to train dataset"}
    )

    valid_dataset_path: Optional[str] = field(
        default="output/data/valid_dataset.pt", metadata={"help": "Path to valid dataset"}
    )


@dataclass
class TrainingArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
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
    
    num_train_epochs: int = field(
        default=4, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    
    save_model: bool = field(default=False, metadata={"help": "Whether to save model."})

    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )

    log_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models"},
    )

def runner(
    train_dataset_path: str,
    valid_dataset_path: Optional[str],
    # class_weight_path: str,
    config_dir: str,
    # model_name_or_path: str,
    cache_dir: Optional[str],
    freeze_feature: bool = False,
    num_train_epochs: str = 2,
    out_dir: str = None,
    log_dir: str = None,
    fp16: bool = False,
    do_train: bool=True,
    do_eval: bool=False
):

    if not os.path.exists(os.path.dirname(log_dir)):
        os.makedirs(os.path.dirname(log_dir))

    config = read_yaml_file(config_dir)["classifier"]    
    model = classifier(
        gcn = False, 
        # pretrained_path=model_args.model_path, 
        pretrained_path=None, 
        freeze_feature = False, 
        n_class=14
    )

    train_dataset = torch.load(train_dataset_path)
    valid_dataset = torch.load(valid_dataset_path) if valid_dataset_path is not None else None
    
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
        do_eval=do_eval
    )
    trainer.train()

    logger.info("Save logs at directory: %s", log_dir)
    # logger.info("Save class weight at directory %s", class_weight_path)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    runner(
        train_dataset_path=data_args.train_dataset_path,
        valid_dataset_path=data_args.valid_dataset_path,
        # class_weight_path=data_args.class_weight_path,
        config_dir=model_args.config_dir,
        # model_name_or_path=model_args.model_name_or_path,
        out_dir=model_args.output_dir,
        cache_dir=model_args.cache_dir,
        freeze_feature=model_args.freeze_feature,
        num_train_epochs=training_args.num_train_epochs,
        log_dir=training_args.log_dir,
        fp16=training_args.fp16,
        do_train=training_args.do_train,
        do_eval=training_args.do_eval    
)


if __name__ == "__main__":
    main()
    

# def main(args_file=None):
#     parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

#     if (len(sys.argv) == 2 and sys.argv[1].endswith(".json")) or args_file is not None:
#         # If we pass only one argument to the script and it's the path to a json file,
#         # let's parse it to get our arguments.
#         args_file_path = os.path.abspath(sys.argv[1]) if args_file is None else args_file
#         model_args, data_args, training_args = parser.parse_json_file(json_file=args_file_path)
#     else:
#         model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
#     # Setup logging
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         level=logging.INFO #if training_args.local_rank in [-1, 0] else logging.WARN,
#     )
    
#     logging.warning(
#         "Device: %s, n_gpu: %s",
#         torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
#         torch.cuda.device_count(),
#     )
    
#     # Setup wandb
#     if training_args.report_to is not None:
#         if training_args.report_to == "wandb":
#             wandb.login()
#             wandb.init(
#                 project="chexnet-classifier",
#                 name=model_args.model_name_or_path,
#                 group=model_args.model_type,
#                 tags=["baseline", "chextnet"],
#                 job_type="train",
#             )

#     # Load datasets
#     logging.info("Loading dataset")
#     data = pd.read_csv(data_args.data_path, index_col=0)
#     train_dataloader, val_dataloader  = get_dloader(df=data, fold=0)

#     # Load pretrained model
#     model = classifier(gcn = False, pretrained_path = model_args.model_path, freeze_feature = False, n_class=14)

#     # Initialize our Trainer
#     configs = ConfigTrainer(model_config = model, config_path=training_args.config_path, verbose=None)()
#     trainer = Trainer(
#                 model = model, 
#                 train_data = train_dataloader,
#                 val_data = val_dataloader,  
#                 loss = configs["loss"], 
#                 optimizer = configs["opt"], 
#                 scheduler = configs["scheduler"], 
#                 metric = configs["metric"],
#                 num_train_epochs = training_args.num_train_epochs,
#                 output_dir = training_args.output_dir,
#                 save_model = training_args.save_model,
#                 fp16 = training_args.fp16
#     )
#     logging.getLogger("wandb.run_manager").setLevel(logging.WARNING)

#     # Training
#     if training_args.do_train:
#         trainer.run(mode = ["train"])

#     # Evaluation 
#     if training_args.do_eval:
#         logging.info("*** Evaluation ***")
#         trainer.run(mode = ["valid"])

# if __name__ == "__main__":
#     main()

