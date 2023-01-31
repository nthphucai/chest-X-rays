import torch
import time
import torch.nn as nn
import numpy as np
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler

from ..utils.utils import get_progress
from ..training.losses import __mapping__ as loss_maps
# from classifier.training.models.optimizer import Lookahead

class Trainer:
    def __init__(self,
                repo, 
                model: nn.Module,
                loss: nn.Module,
                optimizer: nn.Module,
                scheduler: nn.Module,
                metric: nn.Module, 
                callbacks: nn.Module,
                path,
                ) :
        
        self.dl_train, self.dl_val = repo
        self.model = model
        self.loss = loss
        self.opt = optimizer
        self.scheduler = scheduler
        self.score = metric
        self.callbacks = callbacks
        self.path = path

        self.metric_name  = "AUC" #metric['name']
        self.subscore     = True #metric['subscore']
  
        self.fold = 0
        # print('fold:', self.fold)

        self.num_epochs = 20
        self.verbose = None
        self.save_logs = False #self.verbose['save_logs']
        self.save_model = False #self.verbose['save_model']

        # _iter: no.of.batch
        self._iter = 1
        self.best_loss = float("inf") 
        self.classes = np.load('/Users/HPhuc/Practice/12. classification/vinbigdata/output/columns_14.npy')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.since = time.time()

    def _loss_and_outputs(self, imgs: torch.Tensor, targets: torch.Tensor):
        imgs    = imgs.to(self.device)
        targets = targets.to(self.device)
        preds   = self.model(imgs)
        
        loss    = self.loss(preds, targets)        
        return loss, preds
    
    def minibatch(self):
      """ check repo + model
      """
      self.model.train()
      imgs, targets = next(iter(self.dl_train)) 
      for epoch in range(self.num_epochs): 
        loss, preds = self._loss_and_outputs(imgs, targets)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        print('loss:', loss.item())

    def _do_train_epoch(self, epoch: int):
      running_loss = 0.0
      subscore    = [0 for i in range(0, len(self.classes))]

      self.model.train() 
      self.opt.zero_grad()
      # scaler = GradScaler()

      for itr, (imgs, targets) in get_progress(enumerate(self.dl_train), total=len(self.dl_train)):
        # with autocast():
        #   loss, preds = self._loss_and_outputs(imgs, targets)

        # scaler.scale(loss).backward()
        # if (itr+1) % self._iter == 0: 
        #   scaler.step(self.opt)
        #   scaler.update()
        #   self.opt.zero_grad()
        
        loss, preds = self._loss_and_outputs(imgs, targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
         
        running_loss += loss.item()
        for i, c in enumerate(self.classes):
          subscore[i] += self.score(preds.detach().cpu(), targets.detach().cpu()).get(f"{c}_{self.metric_name}")
                             
      train_loss = running_loss / len(self.dl_train) 

      subscore = {f"{c}_{self.metric_name}": (subscore[i] / len(self.dl_train)) for i, c in enumerate(self.classes)}
      score = np.mean([subscore.get(str(f"{c}_{self.metric_name}")) for c in self.classes])

      return train_loss , score, subscore
      
    def _do_val_epoch(self, epoch: int): 
      self.model.eval()
      running_loss = 0.0
      subscore = [0 for i in range(0, len(self.classes))]
      
      with torch.no_grad():
        for itr, (imgs, targets) in get_progress(enumerate(self.dl_val), total=len(self.dl_val)):
          loss, preds = self._loss_and_outputs(imgs, targets)
          running_loss += loss.item()
          for i, c in enumerate(self.classes):
            subscore[i] += self.score(preds.detach().cpu(), targets.detach().cpu()).get(f"{c}_{self.metric_name}")

      val_loss = running_loss / len(self.dl_val)  
      subscore = {f"{c}_{self.metric_name}": (subscore[i] / len(self.dl_val)) for i, c in enumerate(self.classes)}
      score = np.mean([subscore.get(str(f"{c}_{self.metric_name}")) for c in self.classes])

      return val_loss, score, subscore

    def run(self, mode = ['train', 'val']):
      for epoch in range(self.num_epochs):
        for phase in mode: 
          if phase == 'train': 
            loss, score, subscore = self._do_train_epoch(epoch) 
            if self.scheduler: self.scheduler.step(loss) if 'val' not in mode else None
            logs = {'train_loss': loss, f'train_{self.metric_name}': score}

          if phase == 'val'  : 
            loss, score, subscore  = self._do_val_epoch(epoch)
            if self.scheduler: self.scheduler.step(loss) 
            logs_ = {'val_loss': loss, 'val_f1': score}
            logs.update(logs_)  

          result = {'Loss': loss, f'{self.metric_name}': score}
          if self.subscore: 
              result.update(subscore)
              logs.update(subscore)

          print('Epoch', f'{epoch}/' + f'{self.num_epochs}', end=' ')
          for key, value in zip(result.keys(), result.values()):
            print(key, ':' , f'{value:4f}', ',' , end=' ') 
          print('\n')

          if self.save_logs:
            csv_file, filename = self.callbacks.save_logs(epoch, logs=logs, name=self.path['log'])

        if self.save_model:
          if (epoch % 1 == 0) and (epoch > 2): 
            #save_loss = float(result.get('Loss'))
            save_loss = float(logs.get('train_loss')) if 'val' not in mode else float(logs.get('val_loss')) 
            if (save_loss < self.best_loss) and (epoch > 2): 
              self.best_loss = save_loss
              self.callbacks.save_model(self.best_loss, epoch, self.model, self.opt, name=self.path['best_model'])   
        
      duration = (time.time() - self.since) / 60
      print('running_time:', f'{duration:.4f}')

      """
      plot results
      """
      if self.verbose:
        self.callbacks.plots(results=filename) 