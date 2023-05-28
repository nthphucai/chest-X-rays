import torch

"""
Normalization
input size: b, c, _, _
"""
def imagenet_norm(img):

  mean = torch.tensor([255*0.485, 255*0.456, 255*0.406]).view(3, 1, 1)
  std  = torch.tensor([255*0.229, 255*0.224, 255*0.225]).view(3, 1, 1)

  img  = (img - mean) / std 
  return img

def min_max_norm(img):
  """
  input images lies in range[0, 255]
  """
  img = img / 255.0
  return img

def mean_std_norm(img, std=127.5, mean=127.5):

  mean = torch.tensor(mean).view(-1, *[1] * len(img.shape[1:]))
  std = torch.tensor(std).view(-1, *[1] * len(img.shape[1:]))

  img = (img - mean) / std
  return img

  
def normalize(img, normaltype= 'imagenet_norm'): 
  if normaltype == 'imagenet_norm':
      img = imagenet_norm(img)
  elif normaltype == 'min_max_norm':
      img = min_max_norm(img)
  elif normaltype == 'mean_std_norm':
      img = mean_std_norm(img)
  return img




