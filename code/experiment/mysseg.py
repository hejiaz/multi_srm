# mystery segment identification experiment code

import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import utils as ut

# mainly used in left-dataset-out experiment, output accuracy of each subject
# arguments:
# transformed_data: a 3d array (nfeature x time x # subjects)
# win_size: window size of time segment matching experiment, default is 9
# return:
# accu: a 1d array (# subjects), accuracy of each subject
def predict(transformed_data, win_size=9):
    ndim, nsample, nsubjs = transformed_data.shape
    accu = np.zeros(shape=nsubjs,dtype=np.float32) 
    nseg = nsample - win_size
    trn_data = np.zeros((ndim*win_size, nseg),order='f',dtype=np.float32)
  
    # the trn data also include the tst data, but will be subtracted when 
    # calculating A
    for m in range(nsubjs):
        for w in range(win_size):
            trn_data[w*ndim:(w+1)*ndim,:] += transformed_data[:,w:(w+nseg),m]
  
    for tst_subj in range(nsubjs):
        tst_data = np.zeros((ndim*win_size, nseg),order='f',dtype=np.float32)
        for w in range(win_size):
            tst_data[w*ndim:(w+1)*ndim,:] = transformed_data[:,w:(w+nseg),tst_subj]  

        corr_mtx = ut.compute_correlation(tst_data.T,(trn_data - tst_data).T)

        for i in range(1,win_size):
            np.fill_diagonal(corr_mtx[:-i,i:],-np.inf)
            np.fill_diagonal(corr_mtx[i:,:-i],-np.inf)
    
        max_result =  np.argmax(corr_mtx, axis=1)
        accu[tst_subj] = sum(max_result == list(range(nseg))) / nseg

    return accu


# used in leave-one-out, output accuracy for left-out subjects
# arguments:
# transformed_data: a 3d array (nfeature x time x # subjects)
# win_size: window size of time segment matching experiment, default is 9
# tst_subj: a list of length (# left-out subjects), index of left-out subjects
# return:
# accu: a 1d array (# left-out subjects), accuracy of left-out subjects
def predict_loo(transformed_data, tst_subj, win_size=9):
    ndim, nsample, nsubjs = transformed_data.shape
    accu = np.zeros(shape=len(tst_subj),dtype=np.float32)  
    nseg = nsample - win_size
    trn_data = np.zeros((ndim*win_size, nseg),order='f',dtype=np.float32)
  
    # the trn data also include the tst data, but will be subtracted when 
    # calculating A
    for m in range(nsubjs):
        for w in range(win_size):
            trn_data[w*ndim:(w+1)*ndim,:] += transformed_data[:,w:(w+nseg),m]
  
    for idx,m in enumerate(tst_subj):
        tst_data = np.zeros((ndim*win_size, nseg),order='f',dtype=np.float32)
        for w in range(win_size):
            tst_data[w*ndim:(w+1)*ndim,:] = transformed_data[:,w:(w+nseg),m]
        
        corr_mtx = ut.compute_correlation(tst_data.T,(trn_data - tst_data).T) 
        for i in range(1,win_size):
            np.fill_diagonal(corr_mtx[:-i,i:],-np.inf)
            np.fill_diagonal(corr_mtx[i:,:-i],-np.inf)
        max_result =  np.argmax(corr_mtx, axis=1)
        accu[idx] = sum(max_result == list(range(nseg))) / nseg

    return accu
