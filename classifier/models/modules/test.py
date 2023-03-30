import numpy as np 
import matplotlib.pyplot as plt
from .commons import display

def test_augs(img: np.array, msk: np.array, augs=None, num_iters=10): 
  """ img size: b, c, h, w
  """

  for i in range(num_iters): 
    if msk is None:
      assert len(img.shape) == 2, 'shape must be (h, w) for augumentation'
      auged = augs(image=img)
      auged_img = auged["image"]
      auged_img = auged_img[np.newaxis, np.newaxis]
      auged_msk = img[np.newaxis, np.newaxis]
 
    else:
      assert len(img.shape) == 4, 'shape must be (b, c, h, w) for display'
      display(idx=0, img1_ls=[img], img2_ls=[msk], name_ls=['orgi'])

      assert len(img.shape) == 4, 'shape must be (b, c, h, w) for augumentation'
      auged = augs({"image": img, "mask": msk})
      auged_img = auged["image"]
      auged_msk = auged["mask"]

    imgs1 = [auged_img]
    imgs2 = [auged_msk]
    name  = ['augs']

    for idx in range(len(imgs1)):
      display(idx=idx, img1_ls=imgs1, img2_ls=imgs2, name_ls=name)