import sys, os
sys.path.insert(0, os.path.abspath('..'))
import numpy as np
import yaml,copy
import utils as ut
from sklearn.utils.extmath import fast_dot
from scipy.stats import rankdata

# map fMRI data to word-embeddings, uses past num_previous time steps in both learning and testing
# use the averaged transformed data, output a single accuracy
# arguments:
# transformed_data: a 3d array (nfeature x time x # subjects in that dataset)
# word_tst: a 2d array (300*(1+num_previous) x time), already performed temporal zero mean and added previous time steps
# W_ft: a 2d array (300*(1+num_previous) x nfeature), linear mapping matrix learned from training data
# num_previous: number of previous time steps added to transformed data and wordembeddings, default is 2
# num_chunks: number of scenes in classification/ranking experiment for each dataset. 
# Chance level is 1/num_chunks
# return:
# accu_class: a 1d array (# datasets), classification accuracy, chance level 1/num_chunks
# accu_rank: a 1d array (# datasets), ranking accuracy, chance level 50%
def predict(transformed_data,word_tst,W_ft,num_chunks,num_previous):
    # Process fmri data
    # average data across all subjects
    ndata = len(transformed_data)
    transformed_avg = []
    for d in range(ndata):       
        nsubj = transformed_data[d].shape[2]
        trans_tmp = transformed_data[d][:,:,0]
        for m in range(1,nsubj):
            trans_tmp += transformed_data[d][:,:,m]
        transformed_avg.append(trans_tmp/nsubj)
    del transformed_data

    FT_classification_score = np.zeros((ndata),dtype=np.float32)
    FT_rank_score = np.zeros((ndata),dtype=np.float32)    
    for d in range(ndata):
        # add previous time steps
        fmri_tst = add_prev_time_steps(transformed_avg[d], num_previous)    
        # comparisons in Semantic space (i.e. fMRI -> text) (procrustes)
        FT_prediction = fast_dot(W_ft, fmri_tst)
        FT_classification_score[d] = scene_classification(word_tst[d], FT_prediction, num_chunks[d])
        FT_rank_score[d] = scene_ranking(word_tst[d], FT_prediction, num_chunks[d])

    # return accu_class,accu_rank
    return FT_classification_score, FT_rank_score

# used in leave-one-out, output accuracy for left-out subjects
# arguments:
# transformed_data: a list of 3d arrays (nfeature x time x # subjects in that dataset)
# word_tst: a list of 2d arrays (300*(1+num_previous) x time), already performed temporal zero mean and added previous time steps
# W_ft: a 2d array (300*(1+num_previous) x nfeature), linear mapping matrix learned from training data
# tst_subj: a list (length-ndata) of lists of length (# left-out subjects in dataset d), index of left-out subjects
# num_previous: number of previous time steps added to transformed data and wordembeddings, default is 2
# num_chunks: a list of length # datasets, number of scenes in classification/ranking experiment for each dataset. 
# Chance level is 1/num_chunks
# return:
# accu_class: a 1d array (# datasets), classification accuracy, chance level 1/num_chunks
# accu_rank: a 1d array (# datasets), ranking accuracy, chance level 50%
def predict_loo(transformed_data,word_tst,W_ft,tst_subj,num_chunks,num_previous):
    # Process fmri data
    # average data across all testing subjects
    ndata = len(transformed_data)
    transformed_avg = []
    for d in range(ndata):       
        nsubj = len(tst_subj[d])
        trans_tmp = transformed_data[d][:,:,tst_subj[d][0]]
        for m in range(1,nsubj):
            trans_tmp += transformed_data[d][:,:,tst_subj[d][m]]
        transformed_avg.append(trans_tmp/nsubj)
    del transformed_data

    FT_classification_score = np.zeros((ndata),dtype=np.float32)
    FT_rank_score = np.zeros((ndata),dtype=np.float32) 
    for d in range(ndata):
        # add previous time steps
        fmri_tst = add_prev_time_steps(transformed_avg[d], num_previous)     
        # comparisons in Semantic space (i.e. fMRI -> text) (procrustes)
        FT_prediction = fast_dot(W_ft, fmri_tst)
        FT_classification_score[d] = scene_classification(word_tst[d], FT_prediction, num_chunks[d])
        FT_rank_score[d] = scene_ranking(word_tst[d], FT_prediction, num_chunks[d])

    # return accu_class,accu_rank
    return FT_classification_score, FT_rank_score

# Learn linear maps: using Procrustes constraint
# learn fMRI -> text (Y -> X) using training data
# transformed train: a list of 2d arrays (feature x nTR(align))
def learn_linear_map(transformed_train,word_tr,num_previous):
    fmri_tr = add_prev_time_steps_all(transformed_train, num_previous)
    W_ft = procrustes_fit(word_tr,fmri_tr)
    return W_ft

# helper function to subtract the mean of the column vectors
def subtract_column_mean(vecs):
    # average the columns
    avg_col_vec = np.mean(vecs,axis=1)
    # subtract the average from the columns
    new_vecs = vecs - avg_col_vec[:,None]
    return new_vecs, avg_col_vec

# data: a 2d array (ndim x nTR)
# prev: num time steps in past
def add_prev_time_steps(data, prev):
    new = copy.copy(data)
    for t in range(prev):
        data = np.pad(data,((0,0),(1,0)),'constant')[:,:-1]
        new = np.concatenate((data,new),axis=0)
    return new

# add previous time steps separately for each dataset and concatenate them along temporal axis
# data: a list (length ndata) of 2d arrays (ndim x nTR)
# new: a 2d array (ndim*(1+prev) x sum of nTR in all datasets)
def add_prev_time_steps_all(data,prev):
    ndim = data[0].shape[0]
    new = np.empty((ndim*(1+prev),0),dtype=np.float32)
    for d in range(len(data)):
        new = np.concatenate((new,add_prev_time_steps(data[d],prev)),axis=1)
    return new

# helper function to perform subtract_column_mean and add_prev_time_steps
# word_tr and word_tst are 2d arrays, (300 x nTR (align/pred))
def process_semantic(word_tr,word_tst,num_previous):
    # Temporal Zero Mean:
    # calculate average for training, and subtract that average out of the test
    word_tr, avg_tr_word_vec = subtract_column_mean(word_tr)
    word_tst = word_tst - avg_tr_word_vec[:, None]
    # add previous time steps to semantic stuff
    word_tr = add_prev_time_steps(word_tr, num_previous)
    word_tst = add_prev_time_steps(word_tst, num_previous)
    return word_tr, word_tst

# process word embeddings of all datasets and concatenate training
# arguments:
# word_tr/word_tst: a list of 2d arrays, (300 x nTR (align/pred))
# return:
# word_tr_all: a 2d array ((1+num_previous)*300 x sum of nTR in all datasets)
# word_tst_all: a list of 2d arrays ((1+num_previous)*100 x nTR in that dataset)
def process_semantic_all(word_tr,word_tst,num_previous):
    ndim = word_tr[0].shape[0]
    word_tr_all = np.empty((ndim*(1+num_previous),0),dtype=np.float32)
    word_tst_all = []
    for d in range(len(word_tr)):
        word_tr_tmp, word_tst_tmp = process_semantic(word_tr[d],word_tst[d],num_previous)
        word_tr_all = np.concatenate((word_tr_all,word_tr_tmp),axis=1)
        word_tst_all.append(word_tst_tmp)
    return word_tr_all,word_tst_all

# We learn the map Y -> X (X = WY)
# where X = voxels x TRs, Y = features x TRs, W = voxels x features 
# since this is orthogonal, the reverse map is given by WT
def procrustes_fit(X,Y):
    A = fast_dot(X,Y.T)
    pert = np.ones(shape=A.shape,dtype=np.float32)
    U, _, VT = np.linalg.svd(A+0.0001*pert, full_matrices=False)
    W = fast_dot(U, VT) # num_voxels x num_features
    return W

# implements scene classification metric
# given predicted fMRI or text, compare with true fMRI or text in the scene classification framework
# prediction is a matrix: dimension x time points
# truth is a matrix: dimension x time points
# num_chunks: number of scenes
# need to average over all scenes
def scene_classification(truth, prediction, num_chunks):
    # chance rate = 1/num_chunks
    scene_length = int(truth.shape[1]/num_chunks)
    # truth matrix
    A = np.reshape(truth[:,:scene_length*num_chunks],newshape=(truth.shape[0]*scene_length,num_chunks),order='F')
    # pred matrix
    B = np.reshape(prediction[:,:scene_length*num_chunks],newshape=(truth.shape[0]*scene_length,num_chunks),order='F')
    corr_mtx = ut.compute_correlation(B.T,A.T)
    # compute accuracy
    max_result =  np.argmax(corr_mtx, axis=1)
    accu = sum(max_result == list(range(num_chunks))) / num_chunks
    return accu


# returns avg rank and vector of predicted ranks 
# implements scene ranking experiment (a 50% probability task)
def scene_ranking(truth, prediction, num_chunks):
    scene_length = int(truth.shape[1]/num_chunks)
    # truth matrix
    A = np.reshape(truth[:,:scene_length*num_chunks],newshape=(truth.shape[0]*scene_length,num_chunks),order='F')
    # pred matrix
    B = np.reshape(prediction[:,:scene_length*num_chunks],newshape=(truth.shape[0]*scene_length,num_chunks),order='F')
    corr_mtx = ut.compute_correlation(B.T,A.T)
    # rank each row (each prediction sample)
    for p in range(num_chunks):
        corr_mtx[p,:] = rankdata(corr_mtx[p,:],method='ordinal')
    # compute average ranking of prediction samples (average of diagonal of correlation matrix)
    score = np.mean(np.diag(corr_mtx))/num_chunks
    return score

