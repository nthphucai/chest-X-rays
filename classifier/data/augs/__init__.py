from .aug2D import MinEdgeResize, ToTorch
from .transforms import train_augs, val_augs

aug_maps = {
    "min_edge_resize": MinEdgeResize,
    "to_torch": ToTorch,
    "train_augs": train_augs,
    "val_augs": val_augs
}