#!/usr/bin/env python

# using ICA (FastICA) for multisubject fMRI data alignment

# do ICA on bX (nsubjs*nvoxel by nTR) concatenate the data vertically

import sys, os
sys.path.insert(0, os.path.abspath('..'))
from utils import mynumpy as np
from scipy import stats
import warnings
# from sklearn.decomposition import FastICA
from .sklearnica import FastICA

def align_voxels(data, niter, nfeature, initseed):
    # "data" is now a list of subjects data
    nvx,nTR = data[0].shape
    nsubjs = len(data)
    for m in range(nsubjs):
        data[m] = np.nan_to_num(data[m])
    # zscore the data
    bX = np.empty(shape=(0,nTR))
    for m in range(nsubjs):
        bX = np.concatenate((bX,np.nan_to_num(stats.zscore(data[m].T ,axis=0, ddof=1).T)),axis=0)
    del data
    bW = []
    # perform ICA
    np.random.seed(initseed)
    A = np.random.rand(nfeature,nfeature).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ica = FastICA(n_components= nfeature, max_iter=200,w_init=A,random_state=initseed)
            St = ica.fit_transform(bX.T)
            ES = St.T
            W = ica.mixing_
            # convert W to list
            for m in range(nsubjs):
                bW.append(W[m*nvx:(m+1)*nvx,:])
        except:
            for m in range(nsubjs):
                bW.append(np.eye(nvx,nfeature))
            ES = bX[:nfeature,:]

    return [], [], bW, [], [], ES
