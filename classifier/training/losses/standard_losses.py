import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _assert_inputs(pred, true):
    assert (
        pred.shape == true.shape
    ), f"predition shape {pred.shape} is not the same as label shape {true.shape}"


class WBCE(nn.Module):
    def __init__(self, weights_path=None, label_smoothing=None, device="cpu"):
        super().__init__()

        if weights_path is None:
            weights = np.array([[1, 1]])
            print("using default weight")
            print(pd.DataFrame(weights, index=["default"]))
        elif ".csv" in weights_path:
            weights = pd.read_csv(weights_path, index_col=0)
            weights = weights.values
        elif ".npy" in weights_path:
            weights = np.load(weights_path)
            print(weights.shape)
        else:
            raise NotImplementedError("only support csv and numpy extension")

        self.weights = torch.tensor(weights, dtype=torch.float, device=device)
        self.lsm = label_smoothing

    def forward(self, preds, trues):
        _assert_inputs(preds, trues)

        ln0 = (1 - preds + 1e-7).log()
        ln1 = (preds + 1e-7).log()

        weights = self.weights
        if self.lsm is not None:
            l1 = weights[..., 0] * (
                (1 - self.lsm) * (1 - trues) * ln0 + (self.lsm / 2) * ln0
            )
            l2 = weights[..., 1] * ((1 - self.lsm) * trues * ln1 + (self.lsm / 2) * ln1)
        else:
            l1 = weights[..., 0] * (1 - trues) * ln0
            l2 = weights[..., 1] * trues * ln1

        loss = l1 + l2
        return -loss.mean()

    def extra_repr(self):
        return f"weights_shape={self.weights.shape}, label_smoothing={self.label_smoothing}, device={self.weights.device}"

    @staticmethod
    def get_class_balance_weight(counts, anchor=0):
        """
        calculate class balance weight from counts with anchor
        :param counts: class counts, shape=(n_class, 2)
        :param anchor: make anchor class weight = 1 and keep the aspect ratio of other weight
        :return: weights for cross entropy loss
        """
        total = counts.values[0, 0] + counts.values[0, 1]
        beta = 1 - 1 / total

        weights = (1 - beta) / (1 - beta**counts)
        normalized_weights = weights / weights.values[:, anchor, np.newaxis]

        return normalized_weights


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, smooth=1e-6):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, preds, label):
        assert (
            preds.shape == label.shape
        ), f"predition shape {preds.shape} is not the same as label shape {label.shape}"

        label = label.float()
        bce = self.bce(preds, label)
        bce = bce.clip(self.smooth, 1.0 - self.smooth)
        pt = torch.exp(-bce)
        focal_bce = bce * (1 - pt) ** self.gamma
        return focal_bce.mean()


class DiceLoss(nn.Module):
    """Calculate dice loss."""

    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num = targets.size(0)
        probability = torch.sigmoid(logits)
        probability = probability.view(num, -1)
        targets = targets.view(num, -1)
        assert probability.shape == targets.shape
        intersection = 2.0 * (probability * targets).sum(axis=-1)
        union = probability.sum(axis=-1) + targets.sum(axis=-1)
        dice_score = (intersection + self.smooth) / union
        return (1.0 - dice_score).mean()


class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2, smooth: float = 1e-6, weight=0.1):
        super(FocalDiceLoss, self).__init__()
        self.focal_loss = FocalLoss(gamma=gamma)
        self.dice_loss = DiceLoss(smooth=smooth)
        self.weight = weight

    def forward(self, preds, label):
        return (
            self.weight * self.focal_loss(preds, label) + self.dice_loss(preds, label)
        ) / (self.weight + 1)


""""
BCE
"""


class BCE(nn.Module):
    def __init__(self, label_smoothing=0.05):
        super().__init__()
        self.lsm = label_smoothing

    def forward(self, preds, trues):
        ln0 = (1 - preds + 1e-6).log()
        ln1 = (preds + 1e-6).log()

        if self.lsm is not None:
            l1 = (1 - self.lsm) * trues * ln1 + (self.lsm / 2) * ln1
            l2 = (1 - self.lsm) * (1 - trues) * ln0 + (self.lsm / 2) * ln0

        else:
            l1 = (1 - trues) * ln0
            l2 = trues * ln1

        loss = l1 + l2

        return -loss.mean().item()
