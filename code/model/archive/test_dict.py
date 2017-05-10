#!/usr/bin/env python
# Check all dataset pairs and output which ones do not have shared subjects
# By Hejia Zhang @ Princeton

import numpy as np
from scipy.stats.mstats import zscore
from DictLearning_orig import MSDL

nsubjs = 6
initseed = 0
nvoxel = 125
nfeature = 20
nTR = 200
niter = 3

loc = np.zeros((125,3),dtype=np.int32)
idx = 0
for i in range(5):
    for j in range(5):
        for k in range(5):
            loc[idx,:] = np.array([i,j,k],dtype=np.int32)
            idx += 1

# random W
W = np.zeros((nvoxel,nfeature,nsubjs))
np.random.seed(initseed)
for m in range(nsubjs):
    A = np.random.rand(nvoxel,nfeature)
    Q, _ = np.linalg.qr(A)
    W[:,:,m] = Q

# random S
S = np.random.rand(nfeature,nTR)

# generate testing data
data = []
for m in range(nsubjs):
	data.append(W[:,:,m].dot(S)+np.random.rand(nvoxel,nTR))

# zscore data
def zscore_data(data):
    return np.nan_to_num(zscore(data.T, axis=0, ddof=1).T)

for m in range(nsubjs):
	data[m] = zscore_data(data[m])

dict_learning = MSDL(factors= nfeature, lam = 1.,rand_seed=initseed, n_iter=niter,method='tvl1')
dict_learning.fit(data,R=loc)





