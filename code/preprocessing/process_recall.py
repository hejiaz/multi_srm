#!/usr/bin/env python
# Preprocess eac masked fmri data to a list of 3d arrays (nvoxel x time x nsubjs)
# By Hejia Zhang @ Princeton

import numpy as np
import yaml
import os,sys
sys.path.insert(0, os.path.abspath('..'))
import pickle
import utils as ut

roi = 'pmc'

# load path
setting = open('../setting.yaml')
options = yaml.safe_load(setting)

# load input data
in_path = options['input_path']+'sherlock/recall_pmc/subj{}.npz'
data = []
for m in range(16):
	ws = np.load(in_path.format(m))
	data_tmp = ut.zscore_data(ws['fmri'].astype(np.float32))
	data.append(data_tmp)

# save results
with open(options['input_path']+'multi_srm/recall_data_all.pickle','wb') as fid:
    pickle.dump(data,fid, pickle.HIGHEST_PROTOCOL)

