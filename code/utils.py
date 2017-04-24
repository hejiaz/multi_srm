#!/usr/bin/env python
# Utilities used in multi-dataset SRM
# By Hejia Zhang @ Princeton

import numpy as np
import pyximport
pyximport.install()
import cython_blas as blas
import sys
import os
import copy
import itertools
from scipy.stats.mstats import zscore
import math
from sklearn.utils.extmath import fast_dot

# zscore data and set NaN to 0
# data: 2d array (voxel x time)
def zscore_data(data):
    return np.nan_to_num(zscore(data.T, axis=0, ddof=1).T)

# zscore data of all subjects and set NaN to 0
# data: 3d array (voxel x time x subj)
def zscore_data_all(data):
    zscored = np.zeros_like(data)
    nsubjs = data.shape[2]
    for m in range(nsubjs):
        zscored[:,:,m] = zscore_data(data[:,:,m])
    return zscored

# find shared subjects between training datasets and left-out dataset. If not shared subject,
# raise an exception
# arguments:
# membership: 2d array (total # subject x total # datasets), the full membership array of 
# all datasets
# loo_ds: indices of left-out dataset
# return:
# idx_loo: a list, (dataset-specific) index of shared subjects in dataset loo_ds
# idx_train: a list, (global in training datasets) index of shared subjects in combined training datasets
def find_shared_subjects_loo_ds(membership,loo_ds):
    nsubjs,ndata = membership.shape
    train_ds = list(range(ndata))
    train_ds.remove(loo_ds)
    train_list = []
    shared_subj_list = []
    idx_loo = []
    idx_train = []
    for m in range(nsubjs):
        # if this subject is valid in training datasets
        train_subj = list(np.squeeze(membership[m,train_ds]))
        if not all(n==-1 for n in train_subj):
            train_list.append(train_subj)
            # if this subject is also in loo_ds
            if membership[m,loo_ds] != -1:
                shared_subj_list.append(train_subj)
                idx_loo.append(membership[m,loo_ds])
    if not idx_loo:
        raise Exception('no shared subjects between training datasets and left-out dataset')
    for m in range(len(idx_loo)):
        idx_train.append(train_list.index(shared_subj_list[m]))
    return idx_loo,idx_train


# find shared subjects between two datasets. If no shared subject, raise an exception
# arguments:
# membership: 2d array (total # subject x total # datasets), the full membership array of 
# all datasets that will be involved in the left-out-dataset experiment
# ds1,ds2: indices of 2 datasets to be compared. 0 to ndata-1
# return:
# idx1, idx2: a list, index of shared subjects in dataset ds1, ds2
def find_shared_subjects_between_two(membership,ds1,ds2):
    idx1 = []
    idx2 = []
    for m in range(membership.shape[0]):
        if membership[m,ds1] != -1 and membership[m,ds2] != -1:
            idx1.append(membership[m,ds1])
            idx2.append(membership[m,ds2])
    if not idx1:
        raise Exception('no shared subject between left-out dataset and base dataset')
    return idx1,idx2

# map index of some subjects in a dataset into global index of these subjects 
# in an (sub) experiment involving datasets ds_all (in the same order)
# arguments:
# membership: 2d array (total # subject x total # datasets), the full membership array of 
# all datasets that will be involved in the full left-out-dataset experiment
# ds: which dataset
# idx_ds: a list, index of some subjects in dataset ds
# ds_all: a list, which datasets are involved in this (sub) experiment
# return:
# idx_all: global index of these subjects in terms of total # subjects in those datasets in the same order
def map_ds_subject_idx_to_global(membership,ds,idx_ds,ds_all):
    # find index of datasets ds in all datasets involved in this (sub) experiment
    ds_all = sorted(ds_all)
    ds_idx = ds_all.index(ds)
    membership = remove_invalid_membership(membership[:,ds_all])
    idx_all = []
    for m in range(len(idx_ds)):
        idx = list(membership[:,ds_idx]).index(idx_ds[m])
        idx_all.append(idx)
    return idx_all

# remove rows and columns with only -1 in a membership array
def remove_invalid_membership(membership):
    nsubjs,ndata = membership.shape
    remove_dataset = []
    for d in range(ndata):
        if all(n==-1 for n in list(membership[:,d])): 
            remove_dataset.append(d)
    membership = np.delete(membership,remove_dataset,axis=1)
    remove_subj = []
    for s in range(nsubjs):
        if all(n==-1 for n in list(membership[s,:])): 
            remove_subj.append(s)
    membership = np.delete(membership,remove_subj,axis=0)
    return membership

# check the membership array to see if the datasets can be linked through common subjects
# important to check train_mb. Not very important for test_mb
# membership: 2d array (total # subjects x total # datasets)
def check_membership(membership):
    nsubjs, ndata = membership.shape
    if ndata == 1:
        return True
    # all shared subjects
    pairs = []
    for m in range(nsubjs):
        ds = list(np.where(membership[m,:] != -1)[0])
        if len(ds) != 1:
            pairs.append(ds)
    # remove duplicates and sort
    pairs.sort()
    pairs = list(pairs for pairs,_ in itertools.groupby(pairs))
    if not pairs:
        return False
    # check if all datasets are linked
    flag=copy.copy(pairs[0])
    for m in range(1,len(pairs)):
        diff = [n for n in pairs[m] if n not in flag]
        if len(diff) == len(pairs[m]) or not diff:
            continue
        else:
            flag.extend(diff)
       
    if sorted(flag) == list(range(ndata)):
        return True 
    else:
        return False

# check if there are at least one testing subjects for each dataset
# test_mb: 2d array (total # (testing) subjects x total # datasets)
def check_test_mb(test_mb):
    ndata = test_mb.shape[1]
    for d in range(ndata):
        if all(n==-1 for n in list(test_mb[:,d])):
            return False
    return True

# count number of subjects in each dataset
# argument:
# membership: 2d array (total # subjects x total # datasets)
# return:
# num_subj: a list, number of subjects in each dataset
def count_num_subject_in_each_dataset(membership):
    ndata = membership.shape[1]
    num_subj = []
    for d in range(ndata):
        num_subj.append(np.count_nonzero(membership[:,d] != -1))
    return num_subj

# find training and testing subjects indices for dataset idx, useful for model 'avg'
# arguments:
# train_mb: a 2d array (# all training subjects x # datasets)
# test_mb: a 2d array (# all testing subjects x # datasets)
# idx: which dataset, can be 0 to ndata-1
# return:
# train_subj: a list, training subjects indices in dataset idx
# test_subj: a list, testing subjects indices in dataset idx
def find_train_test_subj(train_mb,test_mb,idx):
    train_subj = []
    test_subj = []
    for m in range(train_mb.shape[0]):
        if train_mb[m,idx] != -1:
            train_subj.append(train_mb[m,idx])
    for m in range(test_mb.shape[0]):
        if test_mb[m,idx] != -1:
            test_subj.append(test_mb[m,idx])
    return train_subj,test_subj


# learn W matrix of test subjects in a single dataset using data from all datasets
# arguments:
# data: a list of 3d arrays (voxel x time x subjects[d]), each array contains align data (used to compute S)
# from a single dataset. 
# S: a list of 2d arrays (nfeature x time), each array contains shared response from a single dataset
# test_mb: 2d array (# test subjects x total # datasets), membership information of all test subjects
# idx: which dataset to transform, idx can be 0 to ndata-1
# return:
# W: a 3d array (voxel x nfeature x # test subjects in dataset idx)
def learn_test_W_use_all(data,S,test_mb,idx):
    nsubjs,ndata = test_mb.shape
    voxel = data[idx].shape[0]
    nfeature = S[idx].shape[0]
    W = np.empty((voxel,nfeature,0),dtype=np.float32)
    for m in range(nsubjs):
        if test_mb[m,idx] != -1:
            # extract X and S
            X_tmp = np.empty((voxel,0),dtype=np.float32)
            S_tmp = np.empty((nfeature,0),dtype=np.float32)
            for d in range(ndata):
                if test_mb[m,d] != -1:
                    X_tmp = np.concatenate((X_tmp,data[d][:,:,test_mb[m,d]]),axis=1)
                    S_tmp = np.concatenate((S_tmp,S[d]),axis=1)
            # compute W
            Am = fast_dot(X_tmp,S_tmp.T)
            pert = np.eye(voxel,M=nfeature,dtype=np.float32)
            Um, _, Vm = np.linalg.svd(Am+0.0001*pert, full_matrices=False)
            W_tmp = fast_dot(Um,Vm)  # W = UV^T
            W = np.concatenate((W,W_tmp[:,:,None]),axis=2)   
    return W

# learn W matrix of test subjects in a single dataset only using data from this dataset
# arguments:
# data: a list of 3d arrays (voxel x time x subjects[d]), each array contains align data (used to compute S)
# from a single dataset. 
# S: a list of 2d arrays (nfeature x time), each array contains shared response from a single dataset
# test_mb: 2d array (# test subjects x total # datasets), membership information of all test subjects
# idx: which dataset to transform, idx can be 0 to ndata-1
# return:
# W: a 3d array (voxel x nfeature x # test subjects in dataset idx)
def learn_test_W_use_single_dataset(data,S,test_mb,idx):
    nsubjs = test_mb.shape[0]
    voxel = data[idx].shape[0]
    nfeature = S[idx].shape[0]
    W = np.empty((voxel,nfeature,0),dtype=np.float32)
    for m in range(nsubjs):
        if test_mb[m,idx] != -1:
            # compute W
            Am = fast_dot(data[idx][:,:,test_mb[m,idx]],S[idx].T)
            pert = np.eye(voxel,M=nfeature,dtype=np.float32)
            Um, _, Vm = np.linalg.svd(Am+0.0001*pert, full_matrices=False)
            W_tmp = fast_dot(Um,Vm)  # W = UV^T 
            W = np.concatenate((W,W_tmp[:,:,None]),axis=2)
    return W


# learn W for all subjects (both training and testing) in a single dataset (idx)
# subjects in W are in the same order as in data, so that it can be used directly in transform
# Training and testing subjects in train_mb and test_mb are global, not specific for this dataset
# arguments:
# data: a list of 3d arrays (voxel x time x subjects[idx]), each array contains align data (used to compute S)
# from a single dataset. 
# S: a list of 2d arrays (nfeature x time), each array contains shared response from a single dataset
# W: a 3d array (voxel x nfeature x # train subjects), W of training subjects
# train_mb: a 2d array (# train subjects x total # datasets), membership information of all training subjects
# test_mb: a 2d array (# test subjects x total # datasets), membership information of all testing subjects
# idx: learn W for which dataset
# mode: 'all' or 'ind'. Use 'all' for multi_srm and srm_rotate; use 'ind' to degenerate srm_rotate to individual srm
# return:
# W_all: a 3d array (voxel x nfeature x subjects[idx]), subjects are in the same order as in 'data'
# loo: a list, index of test subjects in dataset idx
def learn_W_jointly(data,S,W,train_mb,test_mb,idx,mode):
    # shapes
    voxel,nTR,nsubjs = data[idx].shape
    nfeature = S[idx].shape[0]
    # extract training and testing subjects:
    # 2d arrays (# train/test subjects in dataset idx x 2),[:,0]: index of this subject in train/test subjects;
    # [:,1]: index of this subject in dataset idx
    train_subj = np.empty((0,2),dtype=np.int32)
    test_subj = np.empty((0,2),dtype=np.int32)
    for m in range(train_mb.shape[0]):
        if train_mb[m,idx] != -1:
            train_subj = np.concatenate((train_subj,np.array([[m,train_mb[m,idx]]],dtype=np.int32)),axis=0) 
    for m in range(test_mb.shape[0]):
        if test_mb[m,idx] != -1:
            test_subj = np.concatenate((test_subj,np.array([[m,test_mb[m,idx]]],dtype=np.int32)),axis=0)    
    num_train = train_subj.shape[0]
    num_test = test_subj.shape[0]
    assert (nsubjs == num_train+num_test) 
    # allocate space for W_all
    W_all = np.zeros((voxel,nfeature,nsubjs),dtype=np.float32)
    loo = list(test_subj[:,1])
    # W of training subjects
    for m in range(num_train):
        W_all[:,:,train_subj[m,1]] = W[:,:,train_subj[m,0]]
    # learn W of testing subjects
    if mode == 'all':
        W_test = learn_test_W_use_all(data,S,test_mb,idx)
    elif mode == 'ind':
        W_test = learn_test_W_use_single_dataset(data,S,test_mb,idx)
    else:
        raise Exception('mode of learning W not valid')
    for m in range(num_test):
        W_all[:,:,test_subj[m,1]] = W_test[:,:,m]

    return W_all,loo

# learn W for all subjects (both training and testing) in a single dataset (idx) for algorithm indv_srm
# subjects in W are in the same order as in data, so that it can be used directly in transform
# Training and testing subjects in train_mb and test_mb are global, not specific for this dataset
# arguments:
# data: a list of 3d arrays (voxel x time x subjects[idx]), each array contains align data (used to compute S)
# from a single dataset. 
# S: a list of 2d arrays (nfeature x time), each array contains shared response from a single dataset
# W: a list of 3d array (voxel x nfeature x # train subjects in dataset idx), W of training subjects
# train_mb: a 2d array (# train subjects x total # datasets), membership information of all training subjects
# test_mb: a 2d array (# test subjects x total # datasets), membership information of all testing subjects
# idx: learn W for which dataset
# return:
# W_all: a 3d array (voxel x nfeature x subjects[idx]), subjects are in the same order as in 'data'
# loo: a list, index of test subjects in dataset idx
def learn_W_indv_srm(data,S,W,train_mb,test_mb,idx):
    # shapes
    voxel,nTR,nsubjs = data[idx].shape
    nfeature = S[idx].shape[0]
    # extract training and testing subjects:
    # list (# train/test subjects in dataset idx),index of this subject in dataset idx
    train_subj = []
    test_subj = []
    for m in range(train_mb.shape[0]):
        if train_mb[m,idx] != -1:
            train_subj.append(train_mb[m,idx])
    for m in range(test_mb.shape[0]):
        if test_mb[m,idx] != -1:
            test_subj.append(test_mb[m,idx])    
    num_train = len(train_subj)
    num_test = len(test_subj)
    assert (nsubjs == num_train+num_test) 
    # allocate space for W_all
    W_all = np.zeros((voxel,nfeature,nsubjs),dtype=np.float32)
    loo = test_subj
    # W of training subjects
    for m in range(num_train):
        W_all[:,:,train_subj[m]] = W[idx][:,:,m]
    # learn W of testing subjects
    for m in range(num_test):
        Am = fast_dot(data[idx][:,:,test_subj[m]],S[idx].T)
        pert = np.eye(voxel,M=nfeature,dtype=np.float32)
        Um, _, Vm = np.linalg.svd(Am+0.0001*pert, full_matrices=False)
        W_all[:,:,test_subj[m]] = fast_dot(Um,Vm)  # W = UV^T 
    return W_all,loo


# a wrapper of learning W
# model: 'multi_srm','srm_rotate','srm_rotate_ind','indv_srm'
# see comments of sub-functions for arguments information
def learn_W(data,S,W,train_mb,test_mb,idx,model):
    if model in ['multi_srm','srm_rotate']:
        W_all,loo = learn_W_jointly(data,S,W,train_mb,test_mb,idx,'all')
    elif model in ['srm_rotate_ind']:
        W_all,loo = learn_W_jointly(data,S,W,train_mb,test_mb,idx,'ind')
    elif model in ['indv_srm']:
        W_all,loo = learn_W_indv_srm(data,S,W,train_mb,test_mb,idx)
    else:
        raise Exception('model name not valid')
    return W_all,loo

# learn W of all subjects in the left-out dataset
# arguments:
# data: a 3d array (voxel x time x # subjects in loo_ds)
# S: a 2d array (nfeature x time)
# return:
# W: a 3d array (voxel x nfeature x # subjects in loo_ds)
def learn_W_loo_ds(data,S):
    voxel, _,nsubjs = data.shape
    nfeature = S.shape[0]
    W = np.zeros((voxel,nfeature,nsubjs),dtype=np.float32)
    for m in range(nsubjs):
        Am = fast_dot(data[:,:,m],S.T)
        pert = np.eye(voxel,M=nfeature,dtype=np.float32)
        Um, _, Vm = np.linalg.svd(Am+0.0001*pert, full_matrices=False)
        W[:,:,m] = fast_dot(Um,Vm)  # W = UV^T     
    return W
# transform prediction data of subjects (both training and testing) in a single dataset (idx)
# arguments:
# data: a 3d array (voxel x time x # subjects[idx]), prediction data from dataset idx
# W: a 3d array (voxel x nfeature x # subjects[idx]), subjects are in the same order as in data
# return:
# S: a 3d array (nfeature x time x # subjects[idx])
def transform(data,W):
    nTR = data.shape[1]
    voxel,nfeature,nsubjs = W.shape
    S = np.zeros((nfeature,nTR,nsubjs),dtype=np.float32)
    for m in range(nsubjs):
        S[:,:,m] = zscore_data(fast_dot(W[:,:,m].T,data[:,:,m]))
    return S


def _normalize_for_correlation(data, axis):
    """normalize the data before computing correlation

    The data will be z-scored and divided by sqrt(n)
    along the assigned axis

    Parameters
    ----------
    data: 2D array

    axis: int
        specify which dimension of the data should be normalized

    Returns
    -------
    data: 2D array
        the normalized data
    """
    shape = data.shape
    data = zscore(data, axis=axis, ddof=0)
    # if zscore fails (standard deviation is zero),
    # set all values to be zero
    data = np.nan_to_num(data)
    data = data / math.sqrt(shape[axis])
    return data

def compute_correlation(matrix1, matrix2):
    """compute correlation between two sets of variables

    Correlate the rows of matrix1 with the rows of matrix2.
    If matrix1 == matrix2, it is auto-correlation computation
    resulting in a symmetric correlation matrix.
    The number of columns MUST agree between set1 and set2.
    The correlation being computed here is
    the Pearson's correlation coefficient, which can be expressed as

    .. math:: corr(X, Y) = \\frac{cov(X, Y)}{\\sigma_X\\sigma_Y}

    where cov(X, Y) is the covariance of variable X and Y, and

    .. math:: \\sigma_X

    is the standard deviation of variable X

    Reducing the correlation computation to matrix multiplication
    and using BLAS GEMM API wrapped by Scipy can speedup the numpy built-in
    correlation computation (numpy.corrcoef) by one order of magnitude

    .. math::
        corr(X, Y)
        &= \\frac{\\sum\\limits_{i=1}^n (x_i-\\bar{x})(y_i-\\bar{y})}{(n-1)
        \\sqrt{\\frac{\\sum\\limits_{j=1}^n x_j^2-n\\bar{x}}{n-1}}
        \\sqrt{\\frac{\\sum\\limits_{j=1}^{n} y_j^2-n\\bar{y}}{n-1}}}\\\\
        &= \\sum\\limits_{i=1}^n(\\frac{(x_i-\\bar{x})}
        {\\sqrt{\\sum\\limits_{j=1}^n x_j^2-n\\bar{x}}}
        \\frac{(y_i-\\bar{y})}{\\sqrt{\\sum\\limits_{j=1}^n y_j^2-n\\bar{y}}})

    Parameters
    ----------
    matrix1: 2D array in shape [r1, c]
        MUST be continuous and row-major

    matrix2: 2D array in shape [r2, c]
        MUST be continuous and row-major

    Returns
    -------
    corr_data: 2D array in shape [r1, r2]
        continuous and row-major in np.float32
    """
    matrix1 = matrix1.astype(np.float32)
    matrix2 = matrix2.astype(np.float32)
    [r1, d1] = matrix1.shape
    [r2, d2] = matrix2.shape
    if d1 != d2:
        raise ValueError('Dimension discrepancy')
    # preprocess two components
    matrix1 = _normalize_for_correlation(matrix1, 1)
    matrix2 = _normalize_for_correlation(matrix2, 1)
    corr_data = np.empty((r1, r2),  order='C',dtype=np.float32)
    # blas routine is column-major
    blas.compute_single_matrix_multiplication('T', 'N',
                                              r2, r1, d1,
                                              1.0,
                                              matrix2, d2,
                                              matrix1, d1,
                                              0.0,
                                              corr_data,
                                              r2)
    return corr_data