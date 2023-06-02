import torch
from torch.nn import BCELoss, CrossEntropyLoss, L1Loss, MSELoss

from .standard_losses import WBCE, DiceLoss, FocalDiceLoss, FocalLoss

loss_maps = {
    "wbce": WBCE,
    "ce": CrossEntropyLoss,
    "focal": FocalLoss,
    "bce": BCELoss,
    "mse": MSELoss,
    "dice": DiceLoss,
    "focal_dice": FocalDiceLoss,
}
