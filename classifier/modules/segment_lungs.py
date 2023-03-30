import os 
import gc
from pathlib import Path
from typing import List, Union, Optional
from dataclasses import dataclass, field

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as load_batch
import numpy as np
import pandas as pd
from scipy import ndimage
import albumentations as A
from lungs_segmentation.pre_trained_models import create_model

from classifier.utils.utils import get_progress, multiprocess
from classifier.utils.hf_argparser import HfArgumentParser

@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(
        default="resnet34",
        metadata={"help": "Path to pretrained model."}
    )

    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to data"},
    )

    output_path: Optional[str] = field(
        default="output/lungs-segmentation",
        metadata={"help": "Path to saved output"},
    )

    img_size: Optional[int] = field(
        default=256,
        metadata={"help": "Size of cropped lungs"}
    )

    batch_size: Optional[int] = field(
        default=256,
        metadata={"help": "Batch size"}
    )

class SegmentLung(nn.Module):

    def __init__(self, img_size: int, workers: int, batch_size: int, output_path: str):
        self.output_path = output_path
        self.batch_size = batch_size
        self.workers = workers
        self.img_size = img_size
        self.aug = A.Compose(
            [A.Resize(img_size, img_size, interpolation=1, p=1)], p=1)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def segment(self, model, data, thresh=0.2):
        data_loader_image_path = load_batch(
            dataset=data["img_path"].values, 
            batch_size=self.batch_size, 
            num_workers=self.workers
        ) 
        
        for image_path in data_loader_image_path:
            # try:
            dataset = list(map(self.read_one_image, image_path))
            data_loader = load_batch(
                dataset=dataset, 
                batch_size=self.batch_size, 
                num_workers=self.workers
            )
        
            # except:
            #     continue

            # else:
            model.eval()
            model.to(self.device)
            for image, image_id in get_progress(data_loader):
                image = image.float().to(self.device)
                mask = torch.nn.Sigmoid()(model(image))

                image = image.cpu().detach().numpy()
                mask = mask.cpu().detach().numpy()
                mask = (mask > thresh).astype(np.uint8)
                # merge left and right lung into one
                mask = np.max(mask, axis=1)[:, np.newaxis]
                cropped_lung = map(self._crop_one_image_withLCP, mask, image)
                [np.save(Path(f"{self.output_path}/" + image_id[idc] + ".png"), lung)
                for idc, lung in enumerate(cropped_lung)]
                
                del mask, image, image_id
        
            del dataset, data_loader
            gc.collect()


    def _crop_one_image_withLCP(self, msk: np.ndarray, img: np.ndarray):
        """This function aims to crop img based on the Largest Connected Component.

        Returns:
            numpy array: A cropped image.
        """
        assert all([(len(msk.shape) == 3), (len(img.shape) == 3)]
                   ), "The input shape must be (depth, height, weight)"
        msk = msk[-2:].squeeze(axis=0)
        img = img[0, ...]
        slice_y, slice_x = ndimage.find_objects(msk, 1)[0]

        h, w = slice_y.stop - slice_y.start, slice_x.stop - slice_x.start
        nw, nh = int(w/0.875), int(h/0.875)
        dw, dh = (nw-w)//2, (nh-h)//2

        top = max(slice_y.start-dh, 0)
        left = max(slice_x.start-dw, 0)
        bot = min(slice_y.stop+dh, 1024)
        right = min(slice_x.stop+dw, 1024)
        img_crop = img[top: bot, left: right]

        # Check whether img size > = 256, if not -> padding
        if np.any(np.array(img_crop.shape[:2]) < self.img_size):
            img_crop = self.pad_image(img=img_crop, axes=(
                0, 1), crop_size=[self.img_size])

        augs = self.aug(image=img_crop)
        cropped_image = augs["image"]
        return cropped_image

    @classmethod
    def load_unet_model(cls, model_path_or_name: str):
        unet_model = create_model(model_path_or_name)
        return unet_model

    @staticmethod
    def read_one_image(image_path):
          image_id = os.path.basename(image_path).split(".")[0]
          image = cv2.imread(image_path)
          image = (image - image.min()) / (image.max() - image.min())
          image = image.transpose((2, 0, 1))
          image = torch.from_numpy(image)
          return image, image_id

    @staticmethod
    def pad_image(img, axes: Union[tuple, list], crop_size: int):
        shapes = np.array(img.shape)
        axes = np.array(axes)
        sizes = np.array(shapes[axes])
        diffs = sizes - np.array(crop_size)

        for diff, axis in zip(diffs, axes):
            left = abs(diff) // 2
            right = (left + 1) if diff % 2 != 0 else left
            if diff >= 0:
                continue
            elif diff < 0:
                img = np.pad(img, [(left, right) if i == axis else (
                    0, 0) for i in range(len(shapes))])
        return img

def crop_lung(
    data_path: str,
    model_name_or_path:str,
    output_path:str,
    img_size: int=256,  
    batch_size: int=4, 
):

    data = pd.read_csv(data_path, index_col=0)
    model = SegmentLung.load_unet_model(model_name_or_path)

    segment_lung = SegmentLung(
        img_size=img_size, 
        workers=4, 
        batch_size=batch_size, 
        output_path=output_path
    )

    segment_lung.segment(model=model, data=data)

if __name__ == "__main__":
    parser = HfArgumentParser((Arguments))
    args = parser.parse_args_into_dataclasses()[0]
    
    crop_lung(
        model_name_or_path=args.model_name_or_path, 
        data_path=args.data_path,
        output_path=args.output_path, 
        img_size=args.img_size,
        batch_size=args.batch_size
    )

