import torch
from torch.nn import BCELoss, MSELoss, L1Loss, CrossEntropyLoss

from .standard_losses import DiceLoss, FocalDiceLoss, FocalLoss, WBCE


__mapping__ = {
    "wbce": WBCE,
    "ce": CrossEntropyLoss,
    "focal": FocalLoss,
    "bce": BCELoss,
    "mse": MSELoss,
    "dice": DiceLoss,
    "focal_dice": FocalDiceLoss,
}