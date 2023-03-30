import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def display(idx, img1_ls: list, img2_ls: list, name_ls: list, type='classifier'):
  """ check if input data is list or tuple 
  """ 
  img1_ = img1_ls[idx]
  img2_ = img2_ls[idx]
  name_ = name_ls[idx]
  
  if len(img1_.shape) != 4:
    img1_ = img1_[np.newaxis]
    img2_ = img2_[np.newaxis] 

  assert len(img1_.shape) == 4, 'shape must be (b, c, h, w)'
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))

  ax1.set_title(name_)
  ax1.imshow(img1_[0, img1_.shape[0]//2])

  if type == 'seg': ax2.set_title('seg')
  elif type == 'classifier': ax2.set_title('origin')
  ax2.imshow(img2_[0, img2_.shape[0]//2])
  plt.show()


class GlobalAverage(nn.Module):

    def __init__(self, dims=(-1, -2), keepdims=False):
        """
        :param dims: dimensions of inputs
        :param keepdims: whether to preserve shape after averaging
        """
        super().__init__()
        self.dims = dims
        self.keepdims = keepdims

    def forward(self, x):
        return torch.mean(x, dim=self.dims, keepdim=self.keepdims)

    def extra_repr(self):
        return f"dims={self.dims}, keepdims={self.keepdims}"


class Reshape(nn.Module):

    def __init__(self, *shape):
        """
        final tensor forward result (B, *shape)
        :param shape: shape to resize to except batch dimension
        """
        super().__init__()

        self.shape = shape

    def forward(self, x):
        return x.reshape([x.shape[0], *self.shape])

    def extra_repr(self):
        return f"shape={self.shape}"




