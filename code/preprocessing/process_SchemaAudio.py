#!/usr/bin/env python

import os
import numpy as np
import nibabel as nib
import yaml
from scipy import stats
from nilearn.image import resample_img

# load data path
setting = open('../setting.yaml')
options = yaml.safe_load(setting)

# input and output path
in_path = options['raw_path']+'SchemaAudio/{}/subj{}.nii.gz'
out_path = options['input_path']+'{}/{}/subj{}.npz'
mask_path = options['mask_path']+'{}_3d_mask.nii'

# roi
# roi_all = ['eac','pmc','dmn','pt']
roi_all = ['eac']
# dataset name
# datasets = ['HIMYM','Seinfeld','UpInTheAir','BigBang','Friends','Santa','Shame','Vinny']
datasets = ['HIMYM']
# subject id
# subid = list(range(1,32)) # 31 subjects
subid = [21]
# size info
ws = np.load(options['mask_path']+'wb_3mm_affine.npz')
affine = ws['affine']
wbsize = (61,73,61)

# load mask information
maskdata = []
for roi in roi_all:
    try:
        roimask_fname = mask_path.format(roi)+'.gz'
        mask = nib.load(roimask_fname)
    except:
        roimask_fname = mask_path.format(roi)
        mask = nib.load(roimask_fname)   
    maskdata.append(mask.get_data())
    # create folders
    for ds in datasets:
        if not os.path.exists(options['input_path']+'{}/{}/'.format(ds,roi)):
            os.makedirs(options['input_path']+'{}/{}/'.format(ds,roi))

# loop over different datasets
for ds in datasets:
    # loop over subjects   
    for idx in subid:
        # load bold data
        fname = in_path.format(ds,idx)
        img = nib.load(fname)        
        # resample to MNI 3mm space
        new_img=resample_img(img,target_affine=affine,target_shape=wbsize)
        # extract data
        imgdata = new_img.get_data()
        # loop over different roi
        for r,roi in enumerate(roi_all):
            (i,j,k) = np.where(maskdata[r]>0)
            movie_data = imgdata[i,j,k]
            # zscore data
            movie_data = stats.zscore(movie_data, axis=1)
            # covert NaN to 0
            movie_data = np.nan_to_num(movie_data)
            # save data
            np.savez_compressed(out_path.format(ds,roi,idx-1),fmri=movie_data)
    print (movie_data.shape[1])
    print (out_path.format(ds,0,0))




