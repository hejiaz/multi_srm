#!/usr/bin/env python
# Multi-dataset SRM
# By Hejia Zhang @ Princeton

import numpy as np
import sys
import os
from scipy import stats
from sklearn.utils.extmath import fast_dot

# arguments:
# data: a list of 3d arrays (voxel x time x subjects), each array contains data from a single dataset. 
# Assume data is already z-scored
# membership: a 2d array (total # subjects x total # datasets)
# niter: number of iterations
# nfeature: number of features
# initseed: random seed used to initialize W

# return:
# W: a 3d array (voxel x nfeature x total # subjects)
# S: a list of 2d arrays (nfeature x time), each array contains shared response from a single dataset
def align(data, membership, niter, nfeature, initseed):
    # total data size from 4 datasets (dmn): 493M
    # size information
    nsubjs, ndata = membership.shape # total number of subjects and datasets
    nvoxel = data[0].shape[0] # number of voxels
    nTR = np.zeros((ndata,),dtype=np.int32) # number of TRs of each dataset
    for d in range(ndata):
        nTR[d] = data[d].shape[1]

    # Initialization:
    print ('initialization')
    # Initialize W: Set W[:,:,m] to random orthonormal matrices
    W = np.zeros((nvoxel,nfeature,nsubjs),dtype=np.float32)
    if initseed is not None:
        np.random.seed(initseed)
        for m in range(nsubjs):
            A = np.random.rand(nvoxel,nfeature).astype(np.float32)
            Q, _ = np.linalg.qr(A)
            W[:,:,m] = Q
    else:
        for m in range(nsubjs):
            Q = np.eye(nvoxel,M=nfeature,dtype=np.float32)
            W[:,:,m] = Q
    # Initialize S
    subj = np.zeros((ndata,),dtype=np.int32)
    S = []
    for d in range(ndata):
        S_tmp = np.zeros((nfeature, nTR[d]),dtype=np.float32)
        for m in range(nsubjs):
            if membership[m,d] != -1:
                S_tmp += fast_dot(W[:,:,m].T,data[d][:,:,membership[m,d]])
                subj[d] += 1            
        S_tmp /= subj[d]
        S.append(S_tmp)

    # update
    # obj = np.zeros((niter+1,),dtype=np.float32)
    # obj[0] = obj_val(data,W,S,membership)
    for i in range(niter):
        # print ('iter'+str(i))
        # update W        
        for m in range(nsubjs):
            # extract X and S
            X_tmp = np.empty((nvoxel,0),dtype=np.float32)
            S_tmp = np.empty((nfeature,0),dtype=np.float32)
            for d in range(ndata):
                if membership[m,d] != -1:
                    X_tmp = np.concatenate((X_tmp,data[d][:,:,membership[m,d]]),axis=1)
                    S_tmp = np.concatenate((S_tmp,S[d]),axis=1)
            # compute W
            Am = fast_dot(X_tmp,S_tmp.T)
            pert = np.eye(nvoxel,M=nfeature,dtype=np.float32)
            Um, _, Vm = np.linalg.svd(Am+0.0001*pert, full_matrices=False)
            W[:,:,m] = fast_dot(Um,Vm)  # W = UV^T

        # update S
        for d in range(ndata):
            S_tmp = np.zeros((nfeature, nTR[d]),dtype=np.float32)
            for m in range(nsubjs):
                if membership[m,d] != -1:
                    S_tmp += fast_dot(W[:,:,m].T,data[d][:,:,membership[m,d]])
            S_tmp /= subj[d]
            S[d] = S_tmp

        # # compute objective
        # obj[i+1] = obj_val(data,W,S,membership)
        # print (obj[i+1])
    return W,S


def obj_val(data,W,S,membership):
    obj = 0.
    nsubjs, ndata = membership.shape
    for d in range(ndata):
        for m in range(nsubjs):
            if membership[m,d] != -1:
                obj += np.linalg.norm(data[d][:,:,membership[m,d]]-fast_dot(W[:,:,m],S[d]),ord='fro')**2
    return obj
