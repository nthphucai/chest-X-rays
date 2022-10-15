import pandas as pd
import numpy as np
from abc import abstractmethod
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
import cv2
import torch 
import torch.nn as nn 
from repos.training.data.base_class import StandardDataset
from repos.training.data.normalization import normalize
from repos.training.data.aug2D import ToTorch

class ChextNextDataset(StandardDataset):
  def __init__(self, df, aug, columns):
    super().__init__()
    self.total = df 
    self.aug = aug
    self.columns = columns

  def get_len(self):
    return len(self.total)

  def get_item(self, idx):
    row = self.total.iloc[idx]
    img = np.load(row['Images'])
    label = row[self.columns].values.astype(int)
    # if len(img.shape) > 2:
    #   img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if self.aug is not None: img = self.aug(image=img)['image']
    img, label = ToTorch()(img, label)
    img = torch.FloatTensor(img)
    img = normalize(img, normaltype= 'imagenet_norm')

    return img, label


    
