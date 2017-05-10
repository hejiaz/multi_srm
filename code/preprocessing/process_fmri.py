#!/usr/bin/env python
# Preprocess eac masked fmri data to a list of 3d arrays (nvoxel x time x nsubjs)
# By Hejia Zhang @ Princeton

import numpy as np
import yaml
import os,sys
sys.path.insert(0, os.path.abspath('..'))
import pickle
import utils as ut

# roi = 'pmc'
# nvoxel = 813

# roi = 'pt'
# nvoxel = 1318

# roi = 'eac'
# nvoxel = 1189

roi = 'dmn'
nvoxel = 2329

# load path
setting = open('../setting.yaml')
options = yaml.safe_load(setting)

# load input data
dataset = ['GreenEyes','milky','vodka','sherlock','HIMYM','Seinfeld','UpInTheAir','BigBang','Friends','Santa','Shame','Vinny'] # do not change: same order as the membership array
num_subjs = [40,18,18,16,31,31,31,31,31,31,31,31]
num_TRs = [450,269,269,1973,114,116,126,115,110,118,118,120]
in_path = options['input_path']+'{}/{}/subj{}.npz'
data = []
for nsubjs,ds,nTR in zip(num_subjs,dataset,num_TRs):
	data_tmp = np.zeros((nvoxel,nTR,nsubjs),dtype=np.float32)
	for m in range(nsubjs):
		ws = np.load(in_path.format(ds,roi,m))
		data_tmp[:,:,m] = np.nan_to_num(ut.zscore_data(ws['fmri']))
	data.append(data_tmp)

# save results
with open(options['input_path']+'multi_srm/{}_data_all.pickle'.format(roi),'wb') as fid:
    pickle.dump(data,fid, pickle.HIGHEST_PROTOCOL)

