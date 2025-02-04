#!/usr/bin/env python
# Check all dataset pairs and output which ones do not have shared subjects
# By Hejia Zhang @ Princeton

import numpy as np
from scipy.stats.mstats import zscore
from dictlearn import align

ndata = 3
nsubjs = 6
initseed = 0
nvoxel = 125
nfeature = 20
nTR = [200,60,140]

loc = np.zeros((125,3),dtype=np.int32)
idx = 0
for i in range(5):
    for j in range(5):
        for k in range(5):
            loc[idx,:] = np.array([i,j,k],dtype=np.int32)
            idx += 1

# random W
W = np.zeros((nvoxel,nfeature,nsubjs),dtype=np.float32)
np.random.seed(initseed)
for m in range(nsubjs):
    A = np.random.rand(nvoxel,nfeature).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    W[:,:,m] = Q

# random S
S = []
for d in range(ndata):
	S.append(np.random.rand(nfeature,nTR[d]).astype(np.float32))

# membership
mb_list = [[0,-1,-1],[-1,2,0],[1,-1,-1],[-1,1,-1],[2,0,-1],[3,-1,1]]
# mb_list = [[0,-1,-1],[-1,0,0],[1,-1,-1],[-1,1,-1],[2,2,-1],[3,-1,1]]
membership = np.array(mb_list,dtype=np.int32)
data_subj = [4,3,2]

# generate testing data
data = []
for d in range(ndata):
	data_tmp = np.zeros((nvoxel,nTR[d],data_subj[d]+1),dtype=np.float32)
	for m in range(data_subj[d]):
		Widx = list(membership[:,d]).index(m)
		data_tmp[:,:,m] = W[:,:,Widx].dot(S[d])+np.random.rand(nvoxel,nTR[d]).astype(np.float32)
	data.append(data_tmp)

data[0][:,:,4] = W[:,:,2].dot(S[0])+np.random.rand(nvoxel,nTR[0]).astype(np.float32)
# data[0][:,:,1] = np.zeros((nvoxel,nTR[0]),dtype=np.float32)
membership[2,0] = 4

def zscore_data(data):
    return np.nan_to_num(zscore(data.T, axis=0, ddof=1).T)

# zscore
def zscore_data_all(data):
    zscored = np.zeros_like(data)
    nsubjs = data.shape[2]
    for m in range(nsubjs):
        zscored[:,:,m] = zscore_data(data[:,:,m])
    return zscored

for d in range(ndata):
	data[d] = zscore_data_all(data[d])

W_all, W_new,S_new = align(data,membership,10,nfeature,initseed,'multi_dict',loc)





