import torch
from torch.optim import Adam, SGD


__mapping__ = {
    "adam": Adam,
    "sgd": SGD,
}