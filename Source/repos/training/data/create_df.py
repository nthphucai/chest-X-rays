import os 
import pandas as pd
import numpy as np
import nibabel as nib 

def df_image_mask_path(root):
    
    dataset = []
    remnants = []
    arr_foldername = []

    for folder_name in os.listdir(root):
        arr_foldername.append(folder_name)
        
    for folder_name in sorted(arr_foldername, key = lambda x: str(x.split('_')[-1])):
        arr = []
        nimg = []
        nmask = []
        try:
            for file_name in os.listdir(os.path.join(root, folder_name)):
                found = False
                if 'segmentation' in file_name:
                    for name in nimg:
                        nimg.remove(name)
                        found = True
                        break

                    if found:
                        mask = (nib.load(os.path.join(root, folder_name, file_name)).get_fdata()).sum()
                        containing_mask = 0

                        if mask > 0:
                            containing_mask = 1
                        dataset.append([folder_name,
                                        os.path.join(root, folder_name, name),
                                        os.path.join(root, folder_name, file_name),
                                        containing_mask]) 

                    nmask.append(file_name)


                else:
                    for name in nmask:
                        nmask.remove(name)
                        found = True
                        break
                    
                    if found:
                        mask = (nib.load(os.path.join(root, folder_name, name)).get_fdata()).sum()
                        containing_mask = 0
                        if mask > 0:
                            containing_mask = 1
                        dataset.append([folder_name,
                                        os.path.join(root, folder_name, file_name),
                                        os.path.join(root, folder_name, name), 
                                        containing_mask])
                    nimg.append(file_name)

        except Exception as e:
            pass
        
    return pd.DataFrame(dataset, columns=['case_id', 'img_path', 'seg_path', 'mask'])
    