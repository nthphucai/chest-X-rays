from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...training.callbacks import callback_maps as callback_maps
from ...training.callbacks.utils import save_logs
from ...training.trainer.base_trainer import BaseTrainer
from ...training.trainer.utils import get_dict


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

        self.out_dir = out_dir
        self.log_dir = log_dir
        self.fp16 = fp16

        self.num_train_epochs = num_train_epochs
        self.scaler = torch.cuda.amp.GradScaler()
        self.gradient_accumulation = 1

        self.classes = np.load("data/vinbigdata/vin_classes_14.npy", allow_pickle=True)

    def train_mini_batch(self):
        self.model.train()
        imgs, targets = next(iter(self.dl_train))
        for step in range(self.num_train_epochs):
            loss, _ = self._train_one_batch(step, imgs, targets)
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
        loss, preds = self.loss_and_output(imgs, targets)
        return loss, preds

    def _measures_one_batch(self, preds, targets) -> np.array:
        subscore = [
            self.score(preds.detach().cpu(), targets.detach().cpu()).get(f"{c}")
            for i, c in enumerate(self.classes)
        ]
        subscore = np.array(subscore)
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
                    logs = get_dict(
                        names=["train loss", "auc", *self.classes],
                        values=[loss, score, *subscore],
                        display=True,
                    )

                if m == "eval":
                    loss, score, subscore = self._val_one_epoch(e)
                    self.scheduler.step(loss) if self.scheduler is not None else None
                    logs_ = get_dict(
                        names=["val loss", "auc", *self.classes],
                        values=[loss, score, *subscore],
                        display=True,
                    )
                    logs.update(logs_)

                save_logs(e, logs, self.log_dir) if self.log_dir is not None else None

            [c.on_epoch_end(e, logs) for c in callbacks]

        [c.on_train_end() for c in callbacks]
