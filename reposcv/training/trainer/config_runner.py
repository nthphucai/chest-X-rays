import torch
import torch.nn as nn 
import numpy as np 
import torch.optim as optim

from reposcv.training.losses import __mapping__ as loss_maps
from reposcv.training.optimizers import __mapping__ as opt_maps
from reposcv.training.schedulers import __mapping__ as scheduler_maps
from reposcv.training.metrics import __mapping__ as metric_maps

from reposcv.hub import get_entries

from reposcv.training.trainer.standard_trainer import Trainer
from reposcv.training.data.get_dl import get_dloader
from reposcv.modules.access_files import load_json_file

class ConfigTrainer:

    def __init__(self, 
                model_config: nn.Module,
                config_path: str,
                verbose=None):
        
        training_configs = load_json_file(config_path)
        loss_config = training_configs["loss"] 
        opt_config = training_configs["optimizer"]
        metric_config = training_configs["metric"]
        scheduler_config = training_configs["scheduler"]

        self.model = self._get_model(model_config)
        if verbose:
            num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print('parameters:', f'{num:,}')

        # self.repo = self._get_repo(repo_config)
        # print("creating train, valid loader") if verbose else None

        self.loss = self._get_loss(loss_config)
        print("loss: ", self.loss) if verbose else None

        self.metrics = self._get_metrics(metric_config)
        print("metrics: ", self.metrics) if verbose else None

        self.optimizer = self._get_optimizer(opt_config)
        print("optimizer: ", self.optimizer) if verbose else None

        self.scheduler = self.get_scheduler(scheduler_config)
        print("scheduler: ", self.scheduler) if verbose else None

        # self.callbacks = self._get_callbacks(callbacks_configs)
        # print("callbacks: ", self.callbacks) if verbose else None


    def _get_model(self, model_config):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return model_config.to(device)

    def _get_repo(self, repo_config):
        dl_train, dl_valid = get_dloader(df=repo_config, fold=0)
        return dl_train, dl_valid

    def _get_loss(self, loss_config):
        name = loss_config['name'].lower()
        kwargs = self.get_kwargs(loss_config, excludes=("name", "classes", "path"))

        if "path" in loss_config:
            entries = get_entries(loss_config['path'])
            loss = entries.load_func(loss_config['name'], **kwargs)

        elif name in loss_maps:
            loss = loss_maps[name](**kwargs)
  
        return loss 

    def _get_metrics(self, metric_config):
        name = metric_config["name"].lower()
        kwargs = self.get_kwargs(metric_config, excludes=("name", "path", "subscore"))
        
        if "path" in metric_config:
            entries = get_entries(metric_config['path'])
            metric = entries.load_func(metric_config['name'], **kwargs)

        elif name in metric_maps:
            metric = metric_maps[name](**kwargs)
            
        return metric 

    def _get_optimizer(self, opt_config):
        name = opt_config['name'].lower()
        kwargs = self.get_kwargs(opt_config, excludes=("name", "classes", "path"))

        if name in opt_maps:
            optimizer = opt_maps[name]
        else:
            raise NotImplementedError(f"only support{opt_maps.keys()}")

        optimizer = optimizer([p for p in self.model.parameters() if p.requires_grad], **kwargs)

        return optimizer

    def get_scheduler(self, scheduler_config):
        name = scheduler_config['name'].lower()
        kwargs = self.get_kwargs(scheduler_config, excludes=("name", "classes", "path"))

        if name in scheduler_maps:
            scheduler = scheduler_maps[name]
        else:
            raise NotImplementedError(f"only support{scheduler_maps.keys()}")

        scheduler = scheduler(self.optimizer, **kwargs)

        return scheduler
        
    def _get_callbacks(self, callbacks_config):
        kwargs = self.get_kwargs(callbacks_config, excludes=("name", "path"))

        if "path" in callbacks_config:
            entries = get_entries(callbacks_config['path'])
            callbacks = entries.load_func(callbacks_config['name'], **kwargs)
        
        return callbacks

    def __call__(self):
        return {
            "loss": self.loss,
            "opt": self.optimizer,
            "scheduler": self.scheduler,
            "metric": self.metrics
    }

    @staticmethod
    def get_kwargs(configs, excludes=("name", )):
        return {k: configs[k] for k in configs if k not in excludes}
