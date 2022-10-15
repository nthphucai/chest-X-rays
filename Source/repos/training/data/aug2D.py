import cv2 
import numpy as np
import pandas as pd 
from albumentations import Compose, OneOf, Resize, HorizontalFlip, RandomGamma, ShiftScaleRotate, RandomBrightnessContrast, RandomCrop, CenterCrop, Cutout, DualTransform

from parameter import size 

    
"""
https://albumentations.readthedocs.io/en/fix-readthedocs_empty_docs/_modules/albumentations/augmentations/transforms.html
https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/transforms_interface.py#L226
"""
class MinEdgeResize(DualTransform):
    """
    Resize img based on its min edge
    """
    def __init__(self, size, interpolation=cv2.INTER_LINEAR, always_apply=True, p=0.5):
        """
        size : final size of min edge
        p (float): probability of applying the transform.
        return: resized image
        """
        super().__init__(always_apply, p)
        self.size = size
        self.interpolation = interpolation

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        h, w = img.shape[:2]
        min_edge = min(h, w)

        size = self.size
        new_h = np.round(h / min_edge * size).astype(int)
        new_w = np.round(w / min_edge * size).astype(int)
        img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

        return img

    def get_params(self):
        return {"interpolation": self.interpolation}

"""
Augumentatiion: ToTorch()
"""
from albumentations import DualTransform, ImageOnlyTransform

# class ToTorch(DualTransform):
#   def __init__(self, always_apply=True):
#     super().__init__(always_apply=True)

#   def apply(self, img, **params):
#     assert len(img.shape) in {2,3}, f"image shape must either be (h, w) or (h, w, c), currently {img.shape}"
#     if len(img.shape) == 2:
#       img = img[np.newaxis]
#     else:
#       img = img.transpose([2,0,1])
#     return img


"""
"""
class ToTorch(object):
  import torch
  """
  Change numpy.array(H, W, C) ====> torch.Tensor(C, H, W)
  """
  def __call__(self, img, label):
    assert len(img.shape) in {2,3}, 'the image shape must be either {H,W} or {C,H,W}, current_shape{img.shape}'
    if len(img.shape) == 2:
      img = img[np.newaxis]
    else:
      img = img.transpose([2,0,1])
    return img, label


"""
Sampler
"""
from torch.utils.data import Dataset, RandomSampler, SequentialSampler

class RandomSampler(RandomSampler):

    def __init__(self, dataset):
        super().__init__(dataset, replacement=False)

        self.dataset = dataset

    def __iter__(self):
        self.dataset.reset()

        return super().__iter__()


class SequentialSampler(SequentialSampler):

    def __init__(self, dataset):
        super().__init__(dataset)

        self.dataset = dataset

    def __iter__(self):
        self.dataset.reset()

        return super().__iter__()

train_augs = Compose([
        RandomCrop(size, size), #always_apply=True),
        # MinEdgeResize(size),
        # RandomCrop(size, size), 
        HorizontalFlip(),
        OneOf([
            RandomGamma(p=0.5),
            RandomBrightnessContrast(p=0.5),
            #data.augmentations.HistogramMatching(np.load('hist.npy'), p=0.5),
        ], p=0.3),
        ShiftScaleRotate(shift_limit=0.01, rotate_limit=25, scale_limit=0.1, border_mode=cv2.BORDER_CONSTANT, p=0.3),
    ])

val_augs = Compose([
        RandomCrop(size, size), #always_apply=True),
    ])

