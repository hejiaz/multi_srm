#!/usr/bin/env python
import numpy as np
import yaml
import os
from scipy import stats
import pickle

setting = open('../setting.yaml')
options = yaml.safe_load(setting)

feat_dict = {'multi_srm':[75,75,100],'all_srm':[75,75,50],'indv_srm':[75,75,50],'all_ica':[50,25,25],'indv_ica':[50,25,25],\
'all_gica':[100,50,25],'indv_gica':[100,50,25],'all_dict':[25,25,25],'indv_dict':[25,25,25],'avg':[50,50,50]}
num_train = 40
ds = [0,1,2,3]
rand = [0,1,3,5,6]
expopt_all = ['1st','2nd']
roi_all = ['dmn','pt','eac']

# mysseg
in_path = options['output_path']+'accu/mysseg/{}/{}_feat{}_ntrain{}_rand{}_{}_ds{}.pickle'
out_path = options['output_path']+'accu_bar/mysseg/{}/' # {} is roi
model_all = ['multi_srm','all_srm','indv_srm','all_ica','indv_ica','all_gica','indv_gica','all_dict','indv_dict','avg']
for m,roi in enumerate(roi_all):
    op = out_path.format(roi)
    if not os.path.exists(op):
        os.makedirs(op)     
    for model in model_all:
        nfeat = feat_dict[model]
        accu_all = np.zeros((len(ds),len(rand)),dtype=np.float32)
        for r,initseed in enumerate(rand):
            with open(in_path.format(model,roi,nfeat[m],num_train,initseed,expopt_all[0],ds),'rb') as fid:
                accu1 = pickle.load(fid)
            with open(in_path.format(model,roi,nfeat[m],num_train,initseed,expopt_all[1],ds),'rb') as fid:
                accu2 = pickle.load(fid)
            for d in range(len(ds)):
                accu_tmp = (accu1[d]+accu2[d])/2
                accu_all[d,r] = np.mean(accu_tmp)
        accu_mean = np.mean(accu_all,axis=1)
        accu_se = stats.sem(accu_all,axis=1)
        for d in range(len(ds)):
            np.savez_compressed(op+model+'_ds'+str(ds[d])+'.npz', mean=accu_mean[d],se=accu_se[d])
    print (op)


# mysseg_all
in_path = options['output_path']+'accu/mysseg_all/{}/{}_feat{}_ntrain{}_rand{}_{}_ds{}.pickle'
out_path = options['output_path']+'accu_bar/mysseg_all/{}/' # {} is roi
model_all = ['multi_srm']
for m,roi in enumerate(roi_all):
    op = out_path.format(roi)
    if not os.path.exists(op):
        os.makedirs(op)     
    for model in model_all:
        nfeat = feat_dict[model]
        accu_all = np.zeros((len(ds),len(rand)),dtype=np.float32)
        for r,initseed in enumerate(rand):
            with open(in_path.format(model,roi,nfeat[m],num_train,initseed,expopt_all[0],ds),'rb') as fid:
                accu1 = pickle.load(fid)
            with open(in_path.format(model,roi,nfeat[m],num_train,initseed,expopt_all[1],ds),'rb') as fid:
                accu2 = pickle.load(fid)
            for d in range(len(ds)):
                accu_tmp = (accu1[d]+accu2[d])/2
                accu_all[d,r] = np.mean(accu_tmp)
        accu_mean = np.mean(accu_all,axis=1)
        accu_se = stats.sem(accu_all,axis=1)
        for d in range(len(ds)):
            np.savez_compressed(op+model+'_ds'+str(ds[d])+'.npz', mean=accu_mean[d],se=accu_se[d])
    print (op)