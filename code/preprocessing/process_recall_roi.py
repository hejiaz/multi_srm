#!/usr/bin/env python

# save 

import os
import numpy as np
import nibabel as nib
import yaml
from scipy import stats
import pickle

roi = 'pmc'

# load data path
setting = open('../setting.yaml')
options = yaml.safe_load(setting)
# load mask information
try:
    roimask_fname = options['mask_path']+'{}_3d_mask.nii.gz'.format(roi)
    mask = nib.load(roimask_fname)
except:
    roimask_fname = options['mask_path']+'{}_3d_mask.nii'.format(roi)
    mask = nib.load(roimask_fname) 
maskdata = mask.get_data()
(i,j,k) = np.where(maskdata>0)

# load wb mask information
wbmask_fname = options['mask_path']+'MNI152_T1_3mm_brain_mask.nii'
wbmask = nib.load(wbmask_fname)
wbdata = wbmask.get_data()
ideal_size = np.array([58,58,58])
min_size = np.zeros(3)
max_size = np.zeros(3)
for m in range(3):
    min_size[m] = np.min(np.array(np.where(wbdata > 0))[m,:])
    max_size[m] = np.max(np.array(np.where(wbdata > 0))[m,:]) 
mid_size = (min_size + max_size)/2
min_size = mid_size - ideal_size/2
min_size = min_size.astype(int) + [0,0,3]
max_size = mid_size + ideal_size/2
max_size = max_size.astype(int) + [0,0,4]

# input and output path
in_path = options['raw_path']+'sherlock/image_data_s{}.nii.gz'
out_path = options['input_path']+'sherlock/recall_pmc/subj{}.npz'
if not os.path.exists(options['input_path']+'sherlock/recall_pmc/'):
    os.makedirs(options['input_path']+'sherlock/recall_pmc/')
# subject id
subid = list(range(16)) # 16 subjects

# loop over different subjects
for subj_idx in subid:
    # load bold data
    fname = in_path.format(subj_idx)
    img = nib.load(fname)
    imgdata = img.get_data()
    nTR = imgdata.shape[3]
    newdata = np.zeros((wbdata.shape[0],wbdata.shape[1],wbdata.shape[2],nTR))
    newdata[min_size[0]:max_size[0], min_size[1]:max_size[1], min_size[2]:max_size[2],:] = imgdata
    # extract data
    recall_data = newdata[i,j,k]
    # zscore data
    recall_data = stats.zscore(recall_data, axis=1)
    # covert NaN to 0
    recall_data = np.nan_to_num(recall_data)
    # save data
    np.savez_compressed(out_path.format(subj_idx),fmri=recall_data)
    print (subj_idx)


