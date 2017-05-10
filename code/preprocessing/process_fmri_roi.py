#!/usr/bin/env python

import os
import numpy as np
import nibabel as nib
import yaml
from scipy import stats

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

# input and output path
in_path = options['raw_path']+'{}/'
out_path = options['input_path']+'{}/{}/subj{}.npz'

# dataset-specific settings
# dataset name
datasets = ['GreenEyes','milky','vodka','sherlock']
# subject id
subid_1 = list(range(1,41)) # 40 subjects
subid_2 = list(range(1,19)) # 18 subjects
subid_3 = list(range(19,37)) # 18 subjects
subid_4 = list(range(1,18)) # 16 subjects
subid_4.remove(5)
subid = [subid_1,subid_2,subid_3,subid_4]
# data file name
data_name = ['preprocessed_niftis/sub-{}_task-greenEyes_bold.nii.gz','niftis_preprocessed/subj{}_trans_filtered_func_data.nii','niftis_preprocessed/subj{}_trans_filtered_func_data.nii','sherlock_movie_s{}.nii.gz']
# start and end point
# greeneye: 450 TRs
start1 = [15]*40
end1 = [465]*40
start1[5],start1[27],end1[5],end1[27] = 11,11,461,461
# milky: 269 TRs
start2 = [15]*18
end2 = [284]*18
start2[17],end2[17] = 11,280
# vodka: 269 TRs
start3 = [15]*18
end3 = [284]*18
start3[8],end3[8] = 11,280
# sherlock: 1973 TRs
start4 = [3]*16
end4 = [1976]*16
# put together
start = [start1,start2,start3,start4]
end = [end1,end2,end3,end4]

# loop over different datasets
for (dataset,subj_idx_all,name,st,ed) in zip(datasets,subid,data_name,start,end):
    if not os.path.exists(options['input_path']+'{}/{}/'.format(dataset,roi)):
        os.makedirs(options['input_path']+'{}/{}/'.format(dataset,roi))
    # loop over subjects   
    for sidx, subj_idx in enumerate(subj_idx_all):
        # load bold data
        fname = in_path.format(dataset)+name.format(subj_idx)
        img = nib.load(fname)
        imgdata = img.get_data()
        assert imgdata.shape[0:3] == maskdata.shape
        # extract data
        movie_data = imgdata[i,j,k]
        movie_data = movie_data[:,st[sidx]:ed[sidx]]
        # zscore data
        movie_data = stats.zscore(movie_data, axis=1)
        # covert NaN to 0
        movie_data = np.nan_to_num(movie_data)
        # save data
        np.savez_compressed(out_path.format(dataset,roi,sidx),fmri=movie_data)
    print (out_path.format(dataset,roi,0))

