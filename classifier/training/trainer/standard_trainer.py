import datetime
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from classifier.training.callbacks import __mapping__ as callback_maps
# from classifier.training.callbacks import save_logs
from classifier.training.trainer.base_trainer import BaseTrainer
from classifier.training.trainer.utils import get_dict
from classifier.utils.file_utils import logging


class Trainer(BaseTrainer):
    def __init__(
        self,
        train_data: DataLoader,
        val_data: DataLoader,
        model: nn.Module,
        loss: nn.Module,
        optimizer: nn.Module,
        scheduler: Optional[nn.Module],
        metric: nn.Module,
        num_train_epochs: int,
        out_dir: str = None,
        log_dir: str = None,
        fp16: bool = False,
    ):
        super().__init__(
            model, train_data, val_data, loss, optimizer, scheduler, metric
        )

        self.dl_train = train_data
        self.dl_val = val_data
        self.loss = loss
        self.opt = optimizer
        self.scheduler = scheduler
        self.score = metric
        self.metric_name = "AUC"  # metric['name']

        self.out_dir = out_dir
        self.log_dir = log_dir
        self.fp16 = fp16

        self.num_train_epochs = num_train_epochs
        self.scaler = torch.cuda.amp.GradScaler()
        self.gradient_accumulation = 1

        self.classes = np.load("data/nih/nih_classes_14.npy")

    def train_mini_batch(self):
        self.model.train()
        imgs, targets = next(iter(self.dl_train))
        for iter in range(self.num_train_epochs):
            loss, _ = self._train_one_batch(iter, imgs, targets)
            print("loss:", loss.item())

    def _train_one_batch(self, step, imgs, targets):
        if self.fp16:
            with torch.cuda.amp.autocast():
                loss, preds = self.loss_and_output(imgs, targets)
                self.scaler.scale(loss).backward()
                if (step + 1) % self.gradient_accumulation == 0:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad()
        else:
            loss, preds = self.loss_and_output(imgs, targets)
            self.opt.zero_grad()
            loss.backward()
            if (step + 1) % self.gradient_accumulation == 0:
                self.opt.step()
                self.opt.zero_grad()
        return loss, preds

    def _eval_one_batch(self, imgs, targets):
        loss, preds = self._loss_and_output(imgs, targets)
        return loss, preds

    def _measures_one_batch(self, preds, targets):
        subscore = np.array(
            [
                self.score(preds.detach().cpu(), targets.detach().cpu()).get(
                    f"{c}_{self.metric_name}"
                )
                for i, c in enumerate(self.classes)
            ]
        )
        return subscore

    def run(self, mode=("train", "eval"), callbacks: Union[tuple, list] = None):
        if self.out_dir is not None:
            monitor = "val loss" if "eval" in mode else "train loss"
            model_cp = callback_maps["checkpoint"](
                file_path=self.out_dir, monitor=monitor
            )
            callbacks = callbacks + [model_cp]
        else:
            callbacks = callbacks + []

        [c.set_trainer(self) for c in callbacks]

        train_configs = {
            "train_loader": self.dl_train,
            "test_loader": self.dl_val,
            "start_epoch": 1,
        }

        [c.on_train_begin(**train_configs) for c in callbacks]

        for e in range(self.num_train_epochs):
            print("\nepoch", f"{e}/{self.num_train_epochs}")
            for m in mode:
                if m == "train":
                    [c.on_epoch_begin(e) for c in callbacks]

                    loss, score, subscore = self._train_one_epoch(e, callbacks)
                    result = chain(
                        *[
                            (c, f"{value:4f}")
                            for (c, value) in zip(self.classes, subscore)
                        ]
                    )
                    # print("Loss", loss, *result)

                    logs = get_dict(
                        names=["Loss", f"{self.metric_name}", *self.classes],
                        values=[loss, score, *subscore],
                        display=True,
                    )

                if m == "eval":
                    loss, score, subscore = self._val_one_epoch(e)
                    if self.scheduler is not None:
                        self.scheduler.step(loss)
                    logs_ = get_dict(
                        names=["Loss", f"{self.metric_name}", *self.classes],
                        values=[loss, score, *subscore],
                        display=True,
                    )
                    logs.update(logs_)

                # save_logs(e, logs, self.log_dir) if self.log_dir is not None else None

            [c.on_epoch_end(e, logs) for c in callbacks]

        [c.on_train_end() for c in callbacks]
