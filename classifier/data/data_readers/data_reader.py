import cv2
import numpy as np
import torch

from ..augs import aug_maps
from ..data_readers.standard_class import StandardDataset
from ..normalization import normalize


class ChextNextDataset(StandardDataset):
  def __init__(self, df, augs, columns):
    super().__init__()
    self.total = df 
    self.augs = augs
    self.columns = columns

  def get_len(self):
    return len(self.total)

  def get_item(self, idx):
    row = self.total.iloc[idx]
    img = np.load(row['cropped_img_path'], allow_pickle=True)
    label = row[self.columns].values.astype(int)
    if len(img.shape) > 2:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if self.augs is not None: 
      img = self.augs(image=img)['image']

    img, label = aug_maps['to_torch']()(img, label)
    img = torch.FloatTensor(img)
    img = normalize(img, normaltype= 'imagenet_norm')
    return img, label


    
