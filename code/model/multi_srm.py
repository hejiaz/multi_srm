#!/usr/bin/env python
# Probabilistic version of multi-dataset SRM
# By Hejia Zhang @ Princeton

import numpy as np
import sys,copy
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
# model: should be 'multi_srm'

# return:
# W: a 3d array (voxel x nfeature x total # subjects)
# S: a list of 2d arrays (nfeature x time), each array contains shared response from a single dataset
def align(data, membership, niter, nfeature, initseed, model):
    # size information
    nsubjs_all, ndata = membership.shape # total number of subjects and datasets
    nsubjs = np.zeros((ndata,),dtype=np.int32) # number of subjects of each dataset
    for d in range(ndata):
        nsubjs[d] = sum([a!=-1 for a in list(membership[:,d])])        
    nvoxel = data[0].shape[0] # number of voxels
    nTR = np.zeros((ndata,),dtype=np.int32) # number of TRs of each dataset   
    for d in range(ndata):
        nTR[d] = data[d].shape[1]

    # # only keeps data that will be used here, and adjust membership so that for each dataset,
    # # subject indices are from 0 to nsubjs[d]-1, from top to bottom
    data_new = []
    mem_new = -np.ones_like(membership)
    for d in range(ndata):
        m_d = 0
        data_tmp = np.zeros_like(data[d])
        for m in range(nsubjs_all):
            if membership[m,d] != -1:
                data_tmp[:,:,m_d] = data[d][:,:,membership[m,d]]
                mem_new[m,d] = m_d
                m_d += 1
        data_new.append(data_tmp)
    membership = mem_new
    data = data_new

    bX = []
    bmu = []
    trace_xtx = []
    for d in range(ndata):
        bX_tmp = np.zeros((nsubjs[d]*nvoxel,nTR[d]),dtype=np.float32)
        bmu_tmp = np.zeros((nsubjs[d]*nvoxel),dtype=np.float32)
        trace_xtx_tmp = np.zeros((nsubjs[d]),dtype=np.float32)
        m_d = 0
        for m in range(nsubjs_all):
            if membership[m,d] != -1:
                bmu_tmp[m_d*nvoxel:(m_d+1)*nvoxel] = np.mean(data[d][:,:,membership[m,d]],axis=1)
                trace_xtx_tmp[m_d] = np.sum(data[d][:,:,membership[m,d]] ** 2)
                bX_tmp[m_d*nvoxel:(m_d+1)*nvoxel,:] = data[d][:,:,membership[m,d]]-bmu_tmp[m_d*nvoxel:(m_d+1)*nvoxel][:,None]  
                m_d += 1              
        bX.append(bX_tmp)
        bmu.append(bmu_tmp)
        trace_xtx.append(trace_xtx_tmp)
    del data,bX_tmp,bmu_tmp,trace_xtx_tmp

    # Initialization:
    print ('initialization')
    # Initialize parameters
    bSig_s = []
    sigma2 = []    
    ES = []
    for d in range(ndata):
        bSig_s.append(np.identity(nfeature,dtype=np.float32))
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
        trace_sigma_s = []
        for d in range(ndata):
            # Sum the inverted the sigma2 elements for computing W^T * Psi^-1 * W
            sigma2_tmp = (1 / sigma2[d]).sum()

            # Invert bSig_s[d] using Cholesky factorization (inv_sigma_s)
            (chol_sigma_s, lower_sigma_s) = scipy.linalg.cho_factor(bSig_s[d], check_finite=False)
            inv_sigma_s = scipy.linalg.cho_solve((chol_sigma_s, lower_sigma_s), np.identity(nfeature,dtype=np.float32),check_finite=False)

            # Invert (bSig_s[d] + sigma2_tmp * I) using Cholesky factorization
            sigma_s_rhos = inv_sigma_s + np.identity(nfeature,dtype=np.float32) * sigma2_tmp
            (chol_sigma_s_rhos, lower_sigma_s_rhos) = scipy.linalg.cho_factor(sigma_s_rhos, check_finite=False)
            inv_sigma_s_rhos = scipy.linalg.cho_solve((chol_sigma_s_rhos, lower_sigma_s_rhos),np.identity(nfeature,dtype=np.float32), check_finite=False)

            # Compute the sum of W_i^T * sigma2_i^-2 * X_i, and the sum of traces
            # of X_i^T * sigma2_i^-2 * X_i
            wt_invpsi_x = np.zeros((nfeature, nTR[d]),dtype=np.float32)
            trace_xt_invsigma2_x = 0.0
            for m in range(nsubjs[d]):
                # find W index corresponds to this subject
                m_w = list(membership[:,d]).index(m)
                wt_invpsi_x += (fast_dot(bW[m_w*nvoxel:(m_w+1)*nvoxel,:].T,bX[d][m*nvoxel:(m+1)*nvoxel,:])) / sigma2[d][m]
                trace_xt_invsigma2_x += trace_xtx[d][m] / sigma2[d][m]

            log_det_psi = np.sum(np.log(sigma2[d]) * nvoxel)

            # Update the ES[d]
            ES[d] = fast_dot(fast_dot(bSig_s[d],(np.identity(nfeature,dtype=np.float32)-sigma2_tmp*inv_sigma_s_rhos)),wt_invpsi_x)

            # Update bSig_s[d]  
            bSig_s[d] = inv_sigma_s_rhos + fast_dot(ES[d],ES[d].T) / nTR[d]
            trace_sigma_s.append(nTR[d] * np.trace(bSig_s[d]))

        #    # calculate log likelihood
        #     loglike += loglikelihood(chol_sigma_s_rhos, log_det_psi, chol_sigma_s,trace_xt_invsigma2_x, inv_sigma_s_rhos, wt_invpsi_x,nTR[d])
        # print ('loglike: '+str(loglike))

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
                sigma2[d][m] = trace_xtx[d][m]
                sigma2[d][m] += -2*np.trace(fast_dot(fast_dot(bX[d][m*nvoxel:(m+1)*nvoxel,:].T,bW[m_w*nvoxel:(m_w+1)*nvoxel,:]),ES[d]))
                sigma2[d][m] += trace_sigma_s[d]
                sigma2[d][m] /= (nTR[d] * nvoxel)     


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

    return W,ES,noise


def loglikelihood(chol_sigma_s_rhos, log_det_psi, chol_sigma_s,
            trace_xt_invsigma2_x, inv_sigma_s_rhos, wt_invpsi_x,samples):
    log_det = (np.log(np.diag(chol_sigma_s_rhos) ** 2).sum() + log_det_psi
               + np.log(np.diag(chol_sigma_s) ** 2).sum())
    loglikehood = -0.5 * samples * log_det - 0.5 * trace_xt_invsigma2_x
    loglikehood += 0.5 * np.trace(fast_dot(fast_dot(wt_invpsi_x.T,inv_sigma_s_rhos),wt_invpsi_x))
    return loglikehood


