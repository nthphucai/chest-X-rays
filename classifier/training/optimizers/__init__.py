import torch

from .lookahead import Lookahead

__mapping__ = {"adam": torch.optim.Adam, "look_ahead": Lookahead}
