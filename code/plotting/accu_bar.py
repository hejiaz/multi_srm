#!/usr/bin/env python
import numpy as np
import yaml
import os
from scipy import stats
import pickle

setting = open('../setting.yaml')
options = yaml.safe_load(setting)


# # mapping_old (average across subjects)
# pre = ''
# in_path = options['output_path']+'accu/'+pre+'mapping{}/{}/{}_chunks{}_feat{}_ntrain{}_rand{}_{}_ds{}.npz'
# out_path = options['output_path']+'accu_bar/'+pre+'mapping{}/nfeat{}/num_train{}/ds{}/'

# nfeature = 50
# num_train = 40
# ds = [0,1,2,3]
# word_dim = 300
# rand = [0, 1, 3, 5, 6]
# expopt_all = ['1st','2nd']
# model_all = ['multi_srm','srm_rotate','srm_rotate_ind','indv_srm','avg']
# roi_all = ['dmn','pt','eac']

# for roi in roi_all:
#     accu_class_roi = np.zeros((len(rand),len(model_all)),dtype=np.float32)
#     accu_rank_roi = np.zeros((len(rand),len(model_all)),dtype=np.float32)
#     for m,model in enumerate(model_all):
#         idx = 0
#         for initseed in rand:
#             for expopt in expopt_all:
#                 ws = np.load(in_path.format(word_dim,model,roi,25,nfeature,num_train,initseed,expopt,ds))
#                 accu_class_roi[idx,m] += ws['accu_class']/2
#                 accu_rank_roi[idx,m] += ws['accu_rank']/2
#             idx += 1
#     accu_class_roi_mean = np.mean(accu_class_roi,axis=0)
#     accu_class_roi_se = stats.sem(accu_class_roi,axis=0)
#     accu_rank_roi_mean = np.mean(accu_rank_roi,axis=0)
#     accu_rank_roi_se = stats.sem(accu_rank_roi,axis=0)
#     op = out_path.format(word_dim,nfeature,num_train,ds)
#     if not os.path.exists(op):
#         os.makedirs(op)	    
#     np.savez_compressed(op+roi+'_class.npz', accu_mean=accu_class_roi_mean,accu_se=accu_class_roi_se)
#     np.savez_compressed(op+roi+'_rank.npz', accu_mean=accu_rank_roi_mean,accu_se=accu_rank_roi_se)
#     print (op+roi)
	

# mapping_new (per subject)
pre = ''
in_path = options['output_path']+'accu/'+pre+'mapping{}/{}/{}_chunks{}_feat{}_ntrain{}_rand{}_{}_ds{}_{}.pickle'
out_path = options['output_path']+'accu_bar/'+pre+'mapping{}/nfeat{}/num_train{}/ds{}/'

nfeature = 50
num_train = 40
ds = [0,1,2,3]
word_dim = 300
rand = [0, 1, 3, 5, 6]
expopt_all = ['1st','2nd']
model_all = ['multi_srm','srm_rotate','srm_rotate_ind','indv_srm','avg']
roi_all = ['dmn','pt','eac']

for roi in roi_all:
    accu_roi = []
    accu_roi_mean = np.zeros((len(ds)*len(model_all)),dtype=np.float32)
    accu_roi_se = np.zeros((len(ds)*len(model_all)),dtype=np.float32)
    for d in range(len(ds)):
        accu_roi.append([])
        for m in range(len(model_all)):
            accu_roi[d].append(np.empty((0),dtype=np.float32))
    for m,model in enumerate(model_all):
        for initseed in rand:
            with open(in_path.format(model,roi,nfeature,num_train,initseed,expopt_all[0],ds),'rb') as fid:
                accu1 = pickle.load(fid)
            with open(in_path.format(model,roi,nfeature,num_train,initseed,expopt_all[1],ds),'rb') as fid:
                accu2 = pickle.load(fid)
            for d in range(len(ds)):
                accu_tmp = (accu1[d]+accu2[d])/2
                accu_roi[d][m] = np.concatenate((accu_roi[d][m],np.mean(accu_tmp)[None]))
    idx = 0
    for d in range(len(ds)):
        for m in range(len(model_all)):
            accu_roi_mean[idx] = np.mean(accu_roi[d][m])
            accu_roi_se[idx] = stats.sem(accu_roi[d][m])
            idx += 1
    op = out_path.format(nfeature,num_train,ds)
    if not os.path.exists(op):
        os.makedirs(op)     
    np.savez_compressed(op+roi+'.npz', accu_mean=accu_roi_mean,accu_se=accu_roi_se)
    print (op+roi)


# # mysseg
# in_path = options['output_path']+'accu/mysseg/{}/{}_feat{}_ntrain{}_rand{}_{}_ds{}.pickle'
# out_path = options['output_path']+'accu_bar/mysseg/nfeat{}/num_train{}/ds{}/'

# nfeature = 50
# num_train = 40
# ds = [0,1,2,3]
# rand = [0, 1, 3, 5, 6]
# expopt_all = ['1st','2nd']
# model_all = ['multi_srm','srm_rotate','srm_rotate_ind','indv_srm','avg']
# roi_all = ['dmn','pt','eac']

# for roi in roi_all:
#     accu_roi = []
#     accu_roi_mean = np.zeros((len(ds)*len(model_all)),dtype=np.float32)
#     accu_roi_se = np.zeros((len(ds)*len(model_all)),dtype=np.float32)
#     for d in range(len(ds)):
#         accu_roi.append([])
#         for m in range(len(model_all)):
#             accu_roi[d].append(np.empty((0),dtype=np.float32))
#     for m,model in enumerate(model_all):
#         for initseed in rand:
#             with open(in_path.format(model,roi,nfeature,num_train,initseed,expopt_all[0],ds),'rb') as fid:
#                 accu1 = pickle.load(fid)
#             with open(in_path.format(model,roi,nfeature,num_train,initseed,expopt_all[1],ds),'rb') as fid:
#                 accu2 = pickle.load(fid)
#             for d in range(len(ds)):
#                 accu_tmp = (accu1[d]+accu2[d])/2
#                 accu_roi[d][m] = np.concatenate((accu_roi[d][m],np.mean(accu_tmp)[None]))
#     idx = 0
#     for d in range(len(ds)):
#         for m in range(len(model_all)):
#             accu_roi_mean[idx] = np.mean(accu_roi[d][m])
#             accu_roi_se[idx] = stats.sem(accu_roi[d][m])
#             idx += 1
#     op = out_path.format(nfeature,num_train,ds)
#     if not os.path.exists(op):
#         os.makedirs(op)     
#     np.savez_compressed(op+roi+'.npz', accu_mean=accu_roi_mean,accu_se=accu_roi_se)
#     print (op+roi)


# # mysseg_all
# in_path = options['output_path']+'accu/mysseg_all/{}/{}_feat{}_ntrain{}_rand{}_{}_ds{}.pickle'
# out_path = options['output_path']+'accu_bar/mysseg_all/nfeat{}/num_train{}/ds{}/'

# nfeature = 50
# num_train = 40
# ds = [0,1,2,3]
# rand = [0, 1, 3, 5, 6]
# expopt_all = ['1st','2nd']
# model_all = ['multi_srm','srm_rotate','srm_rotate_ind','indv_srm','avg']
# roi_all = ['dmn','pt','eac']

# for roi in roi_all:
#     accu_roi = []
#     accu_roi_mean = np.zeros((len(ds)*len(model_all)),dtype=np.float32)
#     accu_roi_se = np.zeros((len(ds)*len(model_all)),dtype=np.float32)
#     for d in range(len(ds)):
#         accu_roi.append([])
#         for m in range(len(model_all)):
#             accu_roi[d].append(np.empty((0),dtype=np.float32))
#     for m,model in enumerate(model_all):
#         for initseed in rand:
#             with open(in_path.format(model,roi,nfeature,num_train,initseed,expopt_all[0],ds),'rb') as fid:
#                 accu1 = pickle.load(fid)
#             with open(in_path.format(model,roi,nfeature,num_train,initseed,expopt_all[1],ds),'rb') as fid:
#                 accu2 = pickle.load(fid)
#             for d in range(len(ds)):
#                 accu_tmp = (accu1[d]+accu2[d])/2
#                 accu_roi[d][m] = np.concatenate((accu_roi[d][m],np.mean(accu_tmp)[None]))
#     idx = 0
#     for d in range(len(ds)):
#         for m in range(len(model_all)):
#             accu_roi_mean[idx] = np.mean(accu_roi[d][m])
#             accu_roi_se[idx] = stats.sem(accu_roi[d][m])
#             idx += 1
#     op = out_path.format(nfeature,num_train,ds)
#     if not os.path.exists(op):
#         os.makedirs(op)     
#     np.savez_compressed(op+roi+'.npz', accu_mean=accu_roi_mean,accu_se=accu_roi_se)
#     print (op+roi)


# # loo_ds
# in_path = options['output_path']+'accu/loo_ds/{}/{}_feat{}_rand{}_loo{}_other{}.pickle'
# out_path = options['output_path']+'accu_bar/loo_ds/{}/nfeat{}/'

# nfeature = 50
# model = 'multi_srm'
# # loo_ds_all = [0,0,1,1,1,2,2,3,3,3] 
# # other_ds_all = [[1,3],[3,1],[0,2,3],[0,3,2],[3,0,2],[0,1,3],[0,3,1],[0,1,2],[0,2,1],[1,0,2]]
# loo_ds_all = [0,3,1]
# other_ds_all = [[1,3],[1,0,2],[0,3,2]]

# rand = list(range(5))
# roi_all = ['dmn','pt','eac']
# assert(len(loo_ds_all)==len(other_ds_all))
# num_bar = 0
# for i in range(len(other_ds_all)):
#     num_bar += len(other_ds_all[i])

# for roi in roi_all:
#     accu_roi = []
#     accu_roi_mean = np.zeros((num_bar),dtype=np.float32)
#     accu_roi_se = np.zeros((num_bar),dtype=np.float32)
#     for i in range(num_bar):
#         accu_roi.append(np.empty((0),dtype=np.float32))
#     idx = 0
#     for loo,other in zip(loo_ds_all,other_ds_all):
#         accu_tmp = []        
#         for initseed in rand:
#             with open(in_path.format(model,roi,nfeature,initseed,loo,other),'rb') as fid:
#                 accu_tmp.append(pickle.load(fid))
#         for m in range(len(other)):
#             accu_rand = accu_tmp[0][m]
#             for initseed in range(1,len(rand)):
#                 accu_rand += accu_tmp[initseed][m]
#             accu_rand /= len(rand)
#             accu_roi[idx] = accu_rand
#             idx += 1

#     for idx in range(num_bar):
#         accu_roi_mean[idx] = np.mean(accu_roi[idx])
#         accu_roi_se[idx] = stats.sem(accu_roi[idx])
#     op = out_path.format(model,nfeature)
#     if not os.path.exists(op):
#         os.makedirs(op)     
#     np.savez_compressed(op+roi+'.npz', accu_mean=accu_roi_mean,accu_se=accu_roi_se)
#     print (op+roi)


# # mapping_loo
# pre = ''
# in_path = options['output_path']+'accu/'+pre+'mapping{}_loo/{}/{}_chunks{}_feat{}_rand{}_loods{}.npz'
# out_path = options['output_path']+'accu_bar/'+pre+'mapping{}_loo/nfeat{}/'

# nfeature = 50
# loo_ds_all = [1,2,3]
# word_dim = 300
# rand = list(range(5))
# model_all = ['multi_srm','srm_rotate','srm_rotate_ind','avg']
# roi_all = ['dmn','pt','eac']

# for roi in roi_all:
#     accu_class_roi = np.zeros((len(model_all)*len(loo_ds_all),len(rand)),dtype=np.float32)
#     accu_rank_roi = np.zeros((len(model_all)*len(loo_ds_all),len(rand)),dtype=np.float32)
#     idx = 0
#     for loo_ds in loo_ds_all:
#         for model in model_all:            
#             for r,initseed in enumerate(rand):
#                 ws = np.load(in_path.format(word_dim,model,roi,25,nfeature,initseed,loo_ds))
#                 accu_class_roi[idx,r] = ws['accu_class']
#                 accu_rank_roi[idx,r] = ws['accu_rank']
#             idx += 1
#     accu_class_roi_mean = np.mean(accu_class_roi,axis=1)
#     accu_class_roi_se = stats.sem(accu_class_roi,axis=1)
#     accu_rank_roi_mean = np.mean(accu_rank_roi,axis=1)
#     accu_rank_roi_se = stats.sem(accu_rank_roi,axis=1)
#     op = out_path.format(word_dim,nfeature)
#     if not os.path.exists(op):
#         os.makedirs(op)     
#     np.savez_compressed(op+roi+'_class.npz', accu_mean=accu_class_roi_mean,accu_se=accu_class_roi_se)
#     np.savez_compressed(op+roi+'_rank.npz', accu_mean=accu_rank_roi_mean,accu_se=accu_rank_roi_se)
#     print (op+roi)


