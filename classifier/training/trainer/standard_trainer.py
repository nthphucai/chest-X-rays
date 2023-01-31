import datetime
import torch.optim as optim
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from itertools import chain 
from typing import Iterable

import torch
import torch.nn as nn
import numpy as np
from classifier.training.trainer.utils import get_dict
from classifier.data.callbacks import save_logs, save_model, plots
# from classifier.models.modules.optimizer import Lookahead
from classifier.training.trainer.base_trainer import BaseTrainer
from classifier.utils.file_utils import logging

class Trainer(BaseTrainer):
    def __init__(self, 
                model: nn.Module,
                train_data: Iterable,  
                val_data: Iterable, 
                loss: nn.Module,  
                optimizer: nn.Module, 
                scheduler: nn.Module, 
                metric: nn.Module, 
                num_train_epochs: int,
                output_dir: str, 
                save_model: bool,
                fp16: bool
    ) :
        
        super().__init__(model, train_data, val_data, loss, optimizer, scheduler, metric)

        self.dl_train = train_data
        self.dl_val = val_data        
        self.model = model
        self.loss = loss
        self.opt = optimizer
        self.scheduler = scheduler
        self.score = metric
        self.metric_name  = "AUC" #metric['name']  
        
        self.output_dir = output_dir
        self.save_model = save_model

        self.fold = 0
        self.num_train_epochs = num_train_epochs

        self.best_loss = float("inf") 
        self.classes = np.load('output/columns_14.npy')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
    def train_mini_batch(self):
      self.model.train()
      imgs, targets = next(iter(self.dl_train)) 

      for _ in range(self.num_train_epochs): 
        loss, preds = self._train_one_batch(imgs, targets) 
        print('loss:', loss.item())

    def _loss_and_output(self, imgs, targets):
        imgs    = imgs.to(self.device)
        targets = targets.to(self.device)
        preds   = self.model(imgs)
        loss    = self.loss(preds, targets)
        return loss, preds

    def _train_one_batch(self, imgs, targets):
        loss, preds = self._loss_and_output(imgs, targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss, preds

    def _eval_one_batch(self, imgs, targets):
        loss, preds = self._loss_and_output(imgs, targets)
        return loss, preds

    def _measures_one_batch(self, preds, targets):
        subscore = np.array([self.score(preds.detach().cpu(), targets.detach().cpu()).get(f"{c}_{self.metric_name}") \
                            for i, c in enumerate(self.classes)]
        )
        return subscore

    # def run(self, mode = ['train', 'valid'], callbacks = [LrFinder()]):
    def run(self, mode = ['train', 'valid'], callbacks = []):
        [c.set_trainer(self) for c in callbacks]
        train_configs = {
            "train_loader": self.dl_train,
            "test_loader": self.dl_val,
            "start_epoch": 1,
        }
        [c.on_train_begin(**train_configs) for c in callbacks]

        for e in range(self.num_train_epochs):
            for _ in mode:
                print('Epoch', f'{e}/' + f'{self.num_train_epochs}')
                
                [c.on_epoch_begin(e) for c in callbacks]

                loss, score, subscore = self._train_one_epoch(e, callbacks) 
                result = chain(*[(c, f"{value:4f}") for (c, value) in zip(self.classes, subscore)])
                print("Loss", loss, *result) 

                if self.scheduler: self.scheduler.step(loss) if 'val' not in mode else None

                logs = get_dict(names=['Loss', f'{self.metric_name}', *self.classes], values=[loss, score, *subscore])

                if 'valid' in mode : 
                    loss, score, subscore  = self._val_one_epoch(e)
                    if self.scheduler: self.scheduler.step(loss) 
                    logs_ = get_dict(names=['Loss', f'{self.metric_name}', *self.classes], values=[loss, score, *subscore])
                    logs.update(logs_)  

                if self.output_dir is not None:
                    dates = (datetime.datetime.now()).strftime('%Y%m%d')  
                    remaining_name = 'fold' + str(self.fold) + '_' + dates 

                    log_path = Path(f"{self.output_dir}", "logs", f"{remaining_name}" + ".csv")
                    _, filename = save_logs(e, logs=logs, log_path=log_path)
                    # plots(results=filename) 

                    model_path = Path(f"{self.output_dir}", "checkpoints", f"{remaining_name}" + ".pt")
                    if self.save_model and (e > 1) and (loss < self.best_loss): 
                        self.best_loss = loss
                        save_model(self.best_loss, e, self.model, self.opt, model_path=model_path)   
            
            [c.on_epoch_end(e, logs) for c in callbacks]
        
        logging.info("Save logs at directory: %s", log_path)
        logging.info("Save model at directory: %s", model_path)
        
        [c.on_train_end() for c in callbacks]
