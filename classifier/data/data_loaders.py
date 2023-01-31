import torch.nn as nn 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from parameter import * 
from sklearn.model_selection import KFold
from albumentations import (
    Compose, 
    OneOf, 
    HorizontalFlip, 
    RandomGamma, 
    ShiftScaleRotate, 
    RandomBrightnessContrast, 
    RandomCrop, 
    CenterCrop, 
    Cutout 
)

from ..data.aug2D import train_augs, val_augs
from ..data.data_readers import ChextNextDataset, ToTorch

"""
get repos: get data from repos
"""
class Repos(nn.Module):
  def __init__(self, df, fold=None):
    super().__init__()
    self.fold = fold
    self.df = df

    self.columns = np.load('output/columns_14.npy')

    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    # kfold = MultilabelStratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
    for i, (train_index, val_index) in enumerate(kfold.split(self.df)):
        self.df.loc[val_index, "fold"] = i

  def split_kflod(self, df, fold):
    if fold is not None: 
      train_df = df.loc[df['fold'] != fold].reset_index(drop=True)
      val_df = df.loc[df['fold'] == fold].reset_index(drop=True)

      train_ds = ChextNextDataset(train_df, aug=train_augs, columns=self.columns) 
      val_ds = ChextNextDataset(val_df, aug=val_augs, columns=self.columns) 

    if fold is None:
      train_ds = ChextNextDataset(df, aug=train_augs, columns=self.columns)
      val_ds = None
    return train_ds, val_ds 

  def get_repos(self):
    train_ds, val_ds = self.split_kflod(df=self.df, fold=self.fold)
    return train_ds, val_ds

"""
get loader: load data from repos 
"""
def get_dloader(df, fold, **kwargs):

  ds = Repos(df, fold)
  train_ds, val_ds = ds.get_repos()
  print('data train:', len(train_ds))
  print('data val:', len(val_ds)) if val_ds is not None else None
  train_dl = train_ds.get_loader(**kwargs)
  val_dl = val_ds.get_loader(**kwargs) if val_ds is not None else None

  # Illustrate 
  # idx = np.random.choice(len(train_ds))
  # img, label = train_ds[idx]
  # plt.imshow(img[0], cmap='gray')
  # plt.show()
  # print(img.shape)
  # # print(img)
  # print(label) 

  return train_dl, val_dl
