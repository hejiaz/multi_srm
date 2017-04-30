#!/usr/bin/env python

# using ICA (FastICA) to find a set of dictionary for each dataset or all datasets involved

# do ICA on bX (nsubjs*nvoxel by nTR) concatenate the data vertically

import numpy as np
from .sklearnica import FastICA
from collections import deque
from sklearn.utils.extmath import fast_dot

# arguments:
# data: a list of 3d arrays (voxel x time x # all subjects), each array contains data from a single dataset. 
# Assume data is already z-scored
# membership: a 2d array (total # train subjects x total # datasets)
# niter: number of iterations
# nfeature: number of features
# initseed: random seed used to initialize W
# model: indv_ica or all_ica
# return:
# W: if indv_ica: a list of 3d arrays (voxel x nfeature x # subj[d]), subject indices not aligned with
# indices in 'data', need train_mb to recover this information when doing transformation; 
# if all_ica: a 3d array (voxel x nfeature x # all subjects)
# S: a list of 2d arrays (nfeature x time), each array contains shared response from a single dataset
def align(data, membership, niter, nfeature, initseed, model):
    nsubjs, ndata = membership.shape
    nvoxel = data[0].shape[0]
    nTR = np.zeros((ndata,),dtype=np.int32)
    subj_data = np.zeros((ndata,),dtype=np.int32)
    for d in range(ndata):
        nTR[d] = data[d].shape[1]
        subj_data[d] = np.count_nonzero(membership[:,d] != -1)

    # Separate membership information of each dataset
    # info_list is a list of length ndata. Each element info_list[d] is a 2d array (subj_data[d] x 2)
    # row m of info_list[d]: [idx1, idx2]. Both idx1 and idx2 are index of the m'th subject of dataset d
    # idx1 is the index in all subjects; idx2 is the index in subjects of dataset d
    info_list = []
    for d in range(ndata):
        info_tmp = []
        for m in range(nsubjs):
            if membership[m,d] != -1:
                info_tmp.append([m,membership[m,d]])
        info_list.append(np.array(info_tmp,dtype=np.int32))


    W_raw = []
    S_raw = []
    for d in range(ndata):
        data_tmp = np.empty(shape=(0,nTR[d]),dtype=np.float32)
        for m in range(nsubjs):
            if membership[m,d] != -1:
                data_tmp = np.concatenate((data_tmp,data[d][:,:,membership[m,d]]),axis=0)
        # perform ICA
        np.random.seed(initseed)
        A = np.random.rand(nfeature,nfeature).astype(np.float32)
        ica = FastICA(n_components= nfeature, max_iter=750,w_init=A,random_state=initseed)
        St = ica.fit_transform(data_tmp.T)
        S_raw.append(St.T)
        W_all = ica.mixing_
        W_tmp = np.zeros((nvoxel,nfeature,subj_data[d]),dtype=np.float32)
        for m in range(subj_data[d]):
            W_tmp[:,:,m] = W_all[m*nvoxel:(m+1)*nvoxel,:]
        W_raw.append(W_tmp)

    if model == 'indv_ica':
        return W_raw, S_raw
    elif model == 'all_ica':
        # rotation
        # use first dataset as base
        W_link = W_raw[0]
        info_link = info_list[0][:,0]
        # datasets that are not yet aligned
        not_linked = deque(range(1,ndata))
        while not_linked:
            # find shared subjects between the aligned part and first dataset that is not aligned
            shared,_,diff2 = find_shared(info_link,info_list[not_linked[0]][:,0])
            # if there is no shared subject between them, put this dataset to the end
            if not list(shared):
                not_linked.rotate(-1)
                continue
            else:
                R,W_link,info_link = find_rotation(W_link,W_raw[not_linked[0]],shared,diff2,info_link)
                # rotate W_raw and S_raw
                S_raw[not_linked[0]] = fast_dot(R,S_raw[not_linked[0]])
                not_linked.popleft()    
        # reorder
        W = W_link[:,:,info_link.argsort()]   
        return W, S_raw
    else:
        raise Exception('invalid model')

# find the shared subjects between two datasets. 
# info1 and info2 are the first column of info array of two datasets from info_list.
# arguments:
# info: 1d array (# subjects)
# return:
# shared: 2d array (# shared subjects x 3), [:,0] index in dataset1, [:,1] index in dataset2, [:,2] index in all subjects
# diff1, diff2: 2d array (# remaining subjects x 2), remaining subjects in dataset 1 and 2; 
# [:,0] index in dataset 1/2,[:,1] index in all subjects
def find_shared(info1,info2):
    shared = []
    subjs1 = list(info1)
    subjs2 = list(info2)
    for idx1,subj in enumerate(subjs1):
        try:
            idx2 = subjs2.index(subj)
            shared.append([idx1,idx2,subj])
        except:
            continue
    if not shared:
        return [],[],[]
    shared = np.array(shared)
    diff1_2 = [n for n in subjs1 if n not in list(shared[:,2])]
    diff2_2 = [n for n in subjs2 if n not in list(shared[:,2])]
    diff1_1 = []
    diff2_1 = []
    for subj in diff1_2:
        diff1_1.append(subjs1.index(subj))
    for subj in diff2_2:
        diff2_1.append(subjs2.index(subj))
    # return shared,diff1,diff2
    return shared,np.array([diff1_1,diff1_2]).T,np.array([diff2_1,diff2_2]).T

# find the rotation matrix R that rotate the shared space of dataset2 to dataset1
# W2.dot(R.T).dot(R).dot(S2)
# arguments:
# W1,W2: 3d array (nvoxel x nfeature x subj) for dataset 1 and 2
# shared: shared subjects information from function find_shared
# diff2: remaining subjects information from function find_shared
# info1: the first column of info array of dataset 1 from info_list
# return:
# R: 2d array (nfeature x nfeature), the optimal rotation matrix
# W: 3d array (nvoxel x nfeature x subj(combined)), the new W with combined subjects. For shared subjects, use
# W in dataset 1; for the other subjects in dataset 2, use W2.dot(R.T)
# info: 1d array (# combined subjects), the index of subject m in all subjects
def find_rotation(W1,W2,shared,diff2,info1):
    # find R, where R.T.dot(R)=R.dot(R.T)=I
    # sizes
    nshared = shared.shape[0]
    nfeature = W1.shape[1]
    ndiff2 = diff2.shape[0]
    # aggregate W1 and W2 of shared subjects
    W1_agg = np.empty((nfeature,0),dtype=np.float32)
    W2_agg = np.empty((nfeature,0),dtype=np.float32)
    for m in range(nshared):
        W1_agg = np.concatenate((W1_agg,W1[:,:,shared[m,0]].T),axis=1)
        W2_agg = np.concatenate((W2_agg,W2[:,:,shared[m,1]].T),axis=1)
    # compute R
    Am = fast_dot(W1_agg,W2_agg.T)
    pert = np.eye(nfeature,dtype=np.float32)
    Um, _, Vm = np.linalg.svd(Am+0.0001*pert, full_matrices=False)
    R = fast_dot(Um,Vm)
    # rotate W2 of the remaining subjects in dataset2
    for m in range(ndiff2):
        W2_new = fast_dot(W2[:,:,diff2[m,0]],R.T)
        W1 = np.concatenate((W1,W2_new[:,:,None]),axis=2)
        info1 = np.append(info1,diff2[m,1])
    return R,W1,info1


