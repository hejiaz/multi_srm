#!/usr/bin/env python
# Preprocess word embeddings as a length-ndata list of 2d arrays (ndim x nTR)
# By Hejia Zhang @ Princeton

import numpy as np
import yaml
import pickle

# load path
setting = open('../setting.yaml')
options = yaml.safe_load(setting)

# load input data
ndim = 300
dataset = ['GreenEyes','milky','vodka','sherlock'] # do not change: same order as the membership array
in_path = options['input_path']+'wordembeddings{}/{}.npz'
data = []
for ds in dataset:
	ws = np.load(in_path.format(ndim,ds))
	data.append(ws['text'].astype(np.float32))

# save results
with open(options['input_path']+'multi_srm/word_embedding{}_all.pickle'.format(ndim),'wb') as fid:
    pickle.dump(data,fid, pickle.HIGHEST_PROTOCOL)

