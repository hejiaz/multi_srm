#!/usr/bin/env python

# using ICA (FastICA) to find a set of dictionary for each dataset or all datasets involved

# do ICA on bX (nsubjs*nvoxel by nTR) concatenate the data vertically

import numpy as np
from .DictLearning import MSDL
import scipy,copy
from collections import deque
from sklearn.utils.extmath import fast_dot

# arguments:
# data: a list of 3d arrays (voxel x time x # all subjects), each array contains data from a single dataset. 
# Assume data is already z-scored
# membership: a 2d array (total # train subjects x total # datasets)
# niter: number of iterations
# nfeature: number of features
# initseed: random seed used to initialize W
# model: indv_dict or all_dict
# loc: a 2d array (voxel x 3), 3d location of each voxel
# return:
# W_all: group spatial basis. if indv_dict: a list of 2d arrays (voxel x nfeature); if all_dict: a 2d array (voxel x nfeature)
# W: if indv_dict: a list of 3d arrays (voxel x nfeature x # subj[d]), subject indices not aligned with
# indices in 'data', need train_mb to recover this information when doing transformation; 
# if multi_dict: a 3d array (voxel x nfeature x # all subjects)
# S: a list of 2d arrays (nfeature x time), each array contains shared response from a single dataset
def align(data, membership, niter, nfeature, initseed, model, loc):

    # size information
    nsubjs, ndata = membership.shape # total number of subjects and datasets
    nvoxel = data[0].shape[0] # number of voxels
    nTR = np.zeros((ndata,),dtype=np.int32) # number of TRs of each dataset
    subj_data = np.zeros((ndata,),dtype=np.int32) # number of subjects of each dataset
    for d in range(ndata):
        nTR[d] = data[d].shape[1]
        subj_data[d] = np.count_nonzero(membership[:,d] != -1)

    if model in ['indv_dict']:
        W_all_raw = []
        W_raw = []
        S_raw = []
        for d in range(min([ndata,4])):
        # for d in range(ndata):
            data_tmp = []
            for m in range(nsubjs):
                if membership[m,d] != -1:
                    data_tmp.append(data[d][:,:,membership[m,d]])

            # perform dictionary learning
            dict_learning = MSDL(factors= nfeature, lam=1,rand_seed=initseed, n_iter=niter, method='tvl1')
            dict_learning.fit(data_tmp,R=loc)
            W_tmp = dict_learning.Vs_
            S_tmp = dict_learning.Us_
            W_data = np.zeros((nvoxel,nfeature,subj_data[d]),dtype=np.float32)
            S_data = np.zeros((nfeature,nTR[d]),dtype=np.float32)
            for m in range(subj_data[d]):
                W_data[:,:,m] = W_tmp[m]
                S_data += S_tmp[m].T
            W_all_raw.append(dict_learning.V_)
            W_raw.append(W_data)
            S_raw.append(S_data/subj_data[d])

        for d in range(4,ndata):
            W_all_raw.append([])
            W_raw.append([])
            S_raw.append([])

        return W_all_raw, W_raw, S_raw
    elif model in ['multi_dict']:
        data_tmp = []
        for m in range(nsubjs):
            data_subj = np.empty((nvoxel,0),dtype=np.float32)
            for d in range(ndata):
                if membership[m,d] != -1:
                    data_subj = np.concatenate((data_subj,data[d][:,:,membership[m,d]]),axis=1)
            data_tmp.append(data_subj)
        del data

        # perform dictionary learning
        dict_learning = MSDL(factors= nfeature, lam=1,rand_seed=initseed, n_iter=niter, method='tvl1')
        dict_learning.fit(data_tmp,R=loc)
        W_all = dict_learning.V_
        W_tmp = dict_learning.Vs_
        S_tmp = dict_learning.Us_
        # process results
        W = np.zeros((nvoxel,nfeature,nsubjs),dtype=np.float32)
        for m in range(nsubjs):
            W[:,:,m] = W_tmp[m]
        S = []
        for d in range(ndata):
            S_data = np.zeros((nfeature,nTR[d]),dtype=np.float32)
            S.append(S_data)
        for m in range(nsubjs):
            samples = 0
            for d in range(ndata):
                if membership[m,d] != -1:
                    S[d] += (S_tmp[m][samples:(samples+nTR[d]),:]).T
                    samples += nTR[d]
        for d in range(ndata):
            S[d] = S[d]/subj_data[d]

        return W_all,W, S
    else:
        raise Exception('invalid model')
