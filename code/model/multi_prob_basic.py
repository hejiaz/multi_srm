#!/usr/bin/env python
# Probabilistic version of multi-dataset SRM
# By Hejia Zhang @ Princeton

import numpy as np
import sys
import os
import scipy
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
    # size information
    nsubjs_all, ndata = membership.shape # total number of subjects and datasets
    nsubjs = np.zeros((ndata,),dtype=np.int32) # number of subjects of each dataset
    for d in range(ndata):
        nsubjs[d] = sum([a!=-1 for a in list(membership[:,d])])        
    nvoxel = data[0].shape[0] # number of voxels
    nTR = np.zeros((ndata,),dtype=np.int32) # number of TRs of each dataset   
    for d in range(ndata):
        nTR[d] = data[d].shape[1]
    # only keeps data that will be used here, and adjust membership so that for each dataset,
    # subject indices are from 0 to nsubjs[d]-1, from top to bottom
    # re-arrange the data for each dataset and initialize bmu
    bX = []
    bmu = []
    for d in range(ndata):
        bX_tmp = np.zeros((nsubjs[d]*nvoxel,nTR[d]),dtype=np.float32)
        bmu_tmp = np.zeros((nsubjs[d]*nvoxel),dtype=np.float32)
        membership_d = -np.ones((nsubjs_all,),dtype=np.int32)
        m_d = 0
        for m in range(nsubjs_all):
            if membership[m,d] != -1:
                bX_tmp[m_d*nvoxel:(m_d+1)*nvoxel,:] = data[d][:,:,membership[m,d]]
                bmu_tmp[m_d*nvoxel:(m_d+1)*nvoxel] = np.mean(data[d][:,:,membership[m,d]],axis=1)
                membership_d[m] = m_d
                m_d += 1
        membership[:,d] = membership_d
        bX.append(bX_tmp)
        bmu.append(bmu_tmp)
    del data

    # Initialization:
    print ('initialization')
    # Initialize parameters
    bSig_s = []
    sigma2 = []
    ES = []
    for d in range(ndata):
        bSig_s.append(np.identity(nfeature,dtype=np.float32))
        bmu.append(np.zeros((nvoxel*nsubjs[d]),dtype=np.float32))
        sigma2.append(np.ones((nsubjs[d]),dtype=np.float32))
        ES.append(np.zeros((nfeature,nTR[d]),dtype=np.float32))
    # Initialize W: Set W[:,:,m] to random orthonormal matrices
    bW = np.zeros((nvoxel*nsubjs_all,nfeature),dtype=np.float32)
    if initseed is not None:
        np.random.seed(initseed)
        for m in range(nsubjs_all):
            A = np.random.rand(nvoxel,nfeature).astype(np.float32)
            Q, _ = np.linalg.qr(A)
            bW[m*nvoxel:(m+1)*nvoxel,:] = Q
    else:
        for m in range(nsubjs_all):
            Q = np.eye(nvoxel,M=nfeature,dtype=np.float32)
            bW[m*nvoxel:(m+1)*nvoxel,:] = Q

    # update
    print ('start update')
    for i in range(niter):
        # E-step and some M-step. update ES and bSig_s of each dataset
        loglike = 0.
        for d in range(ndata):
            # remove mean
            bX[d] = bX[d] - bX[d].mean(axis=1)[:,None]
            # find part of bW that is involved in this dataset
            bW_tmp = np.empty((0,nfeature),dtype=np.float32)
            for m in range(nsubjs_all):
                if membership[m,d] != -1: # if m'th subject is in dataset d
                    bW_tmp = np.concatenate((bW_tmp,bW[m*nvoxel:(m+1)*nvoxel,:]),axis=0)
            # compute bSig_x for each dataset, and not saving it
            bSig_x_d = fast_dot(fast_dot(bW_tmp,bSig_s[d]),bW_tmp.T)
            for m in range(nsubjs[d]):
                bSig_x_d[m*nvoxel:(m+1)*nvoxel,m*nvoxel:(m+1)*nvoxel] += sigma2[d][m]*np.identity(nvoxel)
            # update ES[d] and bSig_s[d]
            inv_bSig_x_d = scipy.linalg.inv(bSig_x_d)
            ES[d] = fast_dot(fast_dot(fast_dot(bSig_s[d].T,bW_tmp.T),inv_bSig_x_d),bX[d])
            bSig_s[d] = bSig_s[d] - fast_dot(fast_dot(fast_dot(fast_dot(bSig_s[d].T,bW_tmp.T),inv_bSig_x_d),bW_tmp),bSig_s[d]) + fast_dot(ES[d],ES[d].T)/nTR[d]           
            # calculate log likelihood
            sign , logdet = np.linalg.slogdet(bSig_x_d)
            if sign == -1:
                print (str(i)+'th iteration, log sign negative')
            loglike = loglike -0.5*nTR[d]*logdet - 0.5*np.trace(fast_dot(fast_dot(bX[d].T,inv_bSig_x_d),bX[d])) 
        print ('loglike: '+str(loglike))
        del bSig_x_d, inv_bSig_x_d    

        # M-step: update bW and sigma2    
        for m in range(nsubjs_all):
            # extract bX and ES data associated with this subject
            X_tmp = np.empty((nvoxel,0),dtype=np.float32)
            S_tmp = np.empty((nfeature,0),dtype=np.float32)
            for d in range(ndata):
                if membership[m,d] != -1:
                    m_d = membership[m,d]
                    X_tmp = np.concatenate((X_tmp,bX[d][m_d*nvoxel:(m_d+1)*nvoxel,:]),axis=1)
                    S_tmp = np.concatenate((S_tmp,ES[d]),axis=1)
            # compute bW
            Am = fast_dot(X_tmp,S_tmp.T)
            pert = np.eye(nvoxel,M=nfeature,dtype=np.float32)
            Um, _, Vm = np.linalg.svd(Am+0.0001*pert, full_matrices=False)
            bW[m*nvoxel:(m+1)*nvoxel,:] = fast_dot(Um,Vm)  # W = UV^T
        # update sigma2
        for d in range(ndata):
            for m in range(nsubjs[d]):
                # find out subject index in bW
                m_w = list(membership[:,d]).index(m)
                sigma2[d][m] = np.trace(fast_dot(bX[d][m*nvoxel:(m+1)*nvoxel,:].T,bX[d][m*nvoxel:(m+1)*nvoxel,:]))\
                              -2*np.trace(fast_dot(fast_dot(bX[d][m*nvoxel:(m+1)*nvoxel,:].T,bW[m_w*nvoxel:(m_w+1)*nvoxel,:]),ES[d]))\
                            +nTR[d]*np.trace(bSig_s[d])
                sigma2[d][m] = sigma2[d][m]/(nTR[d]*nvoxel)                

    # reshape bW
    W = np.zeros((nvoxel,nfeature,nsubjs_all),dtype=np.float32)
    for m in range(nsubjs_all):
        W[:,:,m] = bW[m*nvoxel:(m+1)*nvoxel,:]
    del bW
    # noise level
    noise = np.zeros((nsubjs_all,ndata),dtype=np.float32)
    for d in range(ndata):
        for m in range(nsubjs_all):
            if membership[m,d] != -1:
                noise[m,d] = sigma2[d][membership[m,d]]

    return W,ES#,noise

