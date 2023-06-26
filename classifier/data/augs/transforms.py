import cv2
from albumentations import (CenterCrop, Compose, Cutout, DualTransform,
                            HorizontalFlip, OneOf, RandomBrightnessContrast,
                            RandomCrop, RandomGamma, Resize, ShiftScaleRotate)

from parameter import size

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

