import pandas as pd
import numpy as np
import os 
import nibabel as nib
import scipy.ndimage as ndimage
from scipy.ndimage.interpolation import zoom
import parameter as para
from tqdm import tqdm
import shutil
from reposcv.utils import mutiprocess
from sklearn.model_selection import KFold


"""
return 3D volume after preprocess
"""

class preprocess3D:
    def __init__(self, df, name):
        self.df = df
        self.name = name
        self.vol_path = os.path.join(para.output_path, self.name, "{phase}_vol_path".format(phase=self.name))
        self.seg_path = os.path.join(para.output_path, self.name, "{phase}_seg_path".format(phase=self.name))              
        if os.path.exists(self.vol_path) or os.path.exists(self.seg_path):
            shutil.rmtree(self.vol_path)
            shutil.rmtree(self.seg_path)
        print('...make dirs...')
        os.makedirs(self.vol_path)
        os.makedirs(self.seg_path)
        
    def create_npy(self, idx):        
        row = self.df.iloc[idx]
        case_id = row['case_id']
        vol = nib.load(row['img_path'])
        msk = nib.load(row['seg_path'])

        vol_affine = vol.affine[[2, 1, 0, 3]]
        msk_affine = vol.affine[[2, 1, 0, 3]]

        vol = vol.get_fdata()
        msk = msk.get_fdata()
        
        vol = np.clip(vol, para.lower_bound, para.upper_bound)

        new_vol = resample(vol, np.diag(abs(vol_affine)), [3, 1.5, 1.5], order=3)
        new_msk = resample(msk, np.diag(abs(msk_affine)), [3, 1.5, 1.5], order=0)

        crop_vol = crop_pad(new_vol, axes=(1, 2), crop_size=256)
        crop_msk = crop_pad(new_msk, axes=(1, 2), crop_size=256)

        non_zero = new_msk.sum()
        crop_non_zero = crop_msk.sum()

        assert non_zero == crop_non_zero, case_id
    
        np.save(os.path.join(self.vol_path, "{case_id}_imaging".format(case_id=case_id) + '.nii.gz'), crop_vol)
        np.save(os.path.join(self.seg_path, "{case_id}_segmentation".format(case_id=case_id) + '.nii.gz'), crop_msk)
        
        print('processed vol:', crop_vol.shape)
        
    def run(self):
        print('...preprocessing with multiprocess...')
        mutiprocess(self.create_npy, range(len(self.df)))
        self._save_to_df(fold=False)
    
    def _save_to_df(self, fold=False):
        dataset = []
        for file in os.listdir(self.vol_path):
            if file.startswith('case'):
                case = file.split('/')[-1].split('.')[0].split('_')[0]
                _id = file.split('/')[-1].split('.')[0].split('_')[1]
                case_id = case + '_' + _id
                
                vol_file = file
                seg_file = file.replace('imaging', 'segmentation')
                dataset.append([case_id,
                                os.path.join(self.vol_path, vol_file),
                                os.path.join(self.seg_path, seg_file)])

        df = pd.DataFrame(dataset, columns=['case_id', 'new_vol_path', 'new_seg_path'])
        df.to_csv(os.path.join(para.csv_path, self.name + '_ds' + '.csv'))
        
        if fold:
            kf = KFold(n_splits=10, random_state=42, shuffle=True)
            for i, (train_index, val_index) in enumerate(kf.split(df)):
                df.loc[val_index, "fold"] = i
                df.to_csv(os.path.join(para.csv_path, self.name + '_fold.csv'), index=False)
"""
resample(voume, old_shape, new_shape, order)
"""
def resample(v, dxyz, new_dxyz, order=1):
    dz, dy, dx = dxyz[:3]
    new_dz, new_dy, new_dx = new_dxyz[:3]
    
    z, y, x =  v.shape
    
    new_x = np.round(x * dx / new_dx)
    new_y = np.round(y * dy / new_dy)
    new_z = np.round(z * dz / new_dz)
    
    new_v = zoom(v, (new_z / z, new_y / y, new_x / x), order=order)
    return new_v

""" pad image
"""
def pad_image(img, axes, crop_size):
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
          img = np.pad(img, [(left, right) if i == axis else (0, 0) for i in range(len(shapes))])
    return img

"""
crop_pad(volume, axes, crop_sz)
"""    
def crop_pad(v, axes, crop_size=256):
    shapes = np.array(v.shape)
    axes = np.array(axes)
    sizes = np.array(shapes[axes])
    
    diffs = sizes - np.array(crop_size)
    for diff, axis in zip(diffs, axes):
        left = abs(diff) // 2
        right = (left + 1) if diff % 2 != 0 else left
        
        if diff < 0:
            v = np.pad(v, [(left, right) if i == axis else (0, 0) for i in range(len(shapes))])
        elif diff > 0:
            slices = tuple([slice(left, -right) if i == axis else slice(None) for i in range(len(shapes))])
            v = v[slices]
    return v

def crop_pad_z(vol, desired):
    diff = desired - vol.shape[0]
    left = abs(diff) // 2
    right = (left + 1) if diff % 2 != 0 else left

    if diff > 0:
        vol = np.pad(vol, [(left, right), (0, 0), (0, 0)], constant_values=vol.min())
    elif diff < 0:
        slices = tuple([slice(left, -right), slice(None), slice(None)])
        vol = vol[slices]
    return vol

def crop_pad_y(vol, desired):
    diff = desired - vol.shape[1]
    left = abs(diff) // 2
    right = (left + 1) if diff % 2 != 0 else left

    if diff > 0:
        vol = np.pad(vol, [(0, 0), (left, right), (0, 0)], constant_values=vol.min())
    elif diff < 0:
        slices = tuple([slice(None), slice(left, -right), slice(None)])
        vol = vol[slices]
    return vol

def crop_pad_x(vol, desired):
    diff = desired - vol.shape[2]
    left = abs(diff) // 2
    right = (left + 1) if diff % 2 != 0 else left

    if diff > 0:
        vol = np.pad(vol, [(0, 0), (0, 0), (left, right)], constant_values=vol.min())
    elif diff < 0:
        slices = tuple([slice(None), slice(None), slice(left, -right)])
        vol = vol[slices]
    return vol
