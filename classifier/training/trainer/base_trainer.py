from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...utils.utils import get_progress


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        loss: nn.Module,
        optimizer: nn.Module,
        scheduler: nn.Module,
        metric: nn.Module,
    ):
        self.dl_train = train_data
        self.dl_val = val_data
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def loss_and_output(self, imgs, targets):
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)
        preds = self.model(imgs)
        loss = self.loss(preds, targets)
        return loss, preds

    def _train_one_epoch(self, epoch: int, callbacks=[]):
        total_loss = 0.0
        subscore = 0.0

        self.optimizer.zero_grad()
        self.model.train()
        with get_progress(total=len(self.dl_train)) as pbar:
            for step, batch_data in enumerate(self.dl_train):
                data = self._extract_loader(batch_data)
                imgs, targets = data

                [
                    c.on_training_batch_begin(epoch=epoch, step=step, data=data)
                    for c in callbacks
                ]

                loss, preds = self._train_one_batch(step, imgs, targets)
                total_loss += loss.item()
                subscore += self._measures_one_batch(preds, targets)

                pbar.update()
                [
                    c.on_training_batch_end(
                        epoch=epoch, step=step, data=data, logs=total_loss
                    )
                    for c in callbacks
                ]

        train_loss = total_loss / len(self.dl_train)
        subscore, score = self._measures_one_epoch(subscore, self.dl_train)
        return train_loss, score, subscore

    def _val_one_epoch(self, epoch: int, callbacks=[]):
        total_loss = 0.0
        subscore = 0.0

        self.model.eval()
        with torch.no_grad():
            with get_progress(total=len(self.dl_val)) as pbar:
                for step, batch_data in enumerate(self.dl_val):
                    data = self._extract_loader(batch_data)
                    imgs, targets = data

                    [
                        c.on_validation_batch_begin(epoch, step=step, data=data)
                        for c in callbacks
                    ]
                    loss, preds = self._eval_one_batch(imgs, targets)
                    total_loss += loss.item()
                    subscore += self._measures_one_batch(preds, targets)

                    pbar.update()
                    [
                        c.on_validation_batch_end(
                            epoch=epoch, step=step, data=data, logs=total_loss
                        )
                        for c in callbacks
                    ]

        val_loss = total_loss / len(self.dl_val)
        subscore, score = self._measures_one_epoch(subscore, self.dl_val)
        return val_loss, score, subscore

    def _extract_loader(self, batch_data):
        imgs, targets = batch_data
        return imgs, targets

    def _measures_one_epoch(self, subscore, total_data):
        subscore = np.array(subscore) / len(total_data)
        score = np.mean(subscore)
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
