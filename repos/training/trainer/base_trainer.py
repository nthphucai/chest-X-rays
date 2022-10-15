import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from abc import abstractmethod

from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler

from repos.utils import get_progress
from repos.training.models.optimizer import Lookahead

class BaseTrainer:
    def __init__(self,
                model: nn.Module, 
                train_data,
                val_data, 
                loss: nn.Module, 
                optimizer: nn.Module, 
                scheduler: nn.Module, 
                metric: nn.Module, 
    ):        
        self.dl_train =  train_data
        self.dl_val = val_data
        self.model = model
        self.loss = loss
        self.opt = optimizer
        self.scheduler = scheduler
        self.score = metric

    def _train_one_epoch(self, epoch: int):
      total_loss = 0.0
      subscore   = 0.0 
      
      self.opt.zero_grad()
      self.model.train() 
      with get_progress(total=len(self.dl_train)) as pbar:
        for idc, batch_data in enumerate(self.dl_train):
            imgs, targets = self._extract_loader(batch_data)
            loss, preds   = self._train_one_batch(imgs, targets)
        
            pbar.update()

            with torch.no_grad():
                total_loss  += loss.item()
                subscore    += self._measures_one_batch(preds, targets)
                      
      train_loss = total_loss / len(self.dl_train) 
      subscore, score = self._measures_one_epoch(subscore, self.dl_train)

      return train_loss , score, subscore
      
    def _val_one_epoch(self, epoch: int): 
      total_loss = 0.0
      subscore   = 0.0 
      
      self.model.eval()
      with torch.no_grad():
        with get_progress(enumerate(self.dl_val), total=len(self.dl_val)) as pbar:
            for idc, batch_data in enumerate(self.dl_train):
                imgs, targets = self._extract_loader(batch_data)
                loss, preds   = self._eval_one_batch(imgs, targets)
                
                pbar.update()

                total_loss += loss.item()
                subscore   += self._measures_one_batch(preds, targets)

      val_loss = total_loss / len(self.dl_val) 
      self.subscore, score = self._measures_one_epoch(subscore, self.dl_val) 

      return val_loss, score, subscore

    def _extract_loader(self, batch_data):
        imgs, targets = batch_data
        return imgs, targets

    def _measures_one_epoch(self, subscore, total_data):
        subscore = np.array(subscore) / len(total_data)
        score    = np.mean(subscore)
        return subscore, score

    @abstractmethod
    def _measures_one_batch(self, targets, preds):
        pass

    @abstractmethod
    def _train_one_batch(self, imgs, targets):
        pass

    @abstractmethod
    def _eval_one_batch(self, imgs, targets):
        pass



     


             


 