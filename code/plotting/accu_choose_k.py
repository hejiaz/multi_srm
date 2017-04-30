#!/usr/bin/env python
import numpy as np
import yaml
import os
from scipy import stats
import pickle

setting = open('../setting.yaml')
options = yaml.safe_load(setting)

# mysseg
in_path = options['output_path']+'accu/mysseg/{}/{}_feat{}_ntrain{}_rand{}_{}_ds{}.pickle'
out_path = options['output_path']+'accu_k/mysseg/{}/'

nfeature = [50,75,100,125,150,200]
num_train = 40
ds = [0,1,2,3]
rand = [0, 1, 3]
expopt_all = ['1st','2nd']
model_all = ['multi_srm']
roi_all = ['pt','eac']

for model in model_all:
    for roi in roi_all:
        accu_roi = []
        accu_roi_mean = np.zeros((len(ds)*len(nfeature)),dtype=np.float32)
        accu_roi_se = np.zeros((len(ds)*len(nfeature)),dtype=np.float32)
        for d in range(len(ds)):
            accu_roi.append([])
            for m in range(len(nfeature)):
                accu_roi[d].append(np.empty((0),dtype=np.float32))
        for m,feat in enumerate(nfeature):
            for initseed in rand:
                with open(in_path.format(model,roi,feat,num_train,initseed,expopt_all[0],ds),'rb') as fid:
                    accu1 = pickle.load(fid)
                with open(in_path.format(model,roi,feat,num_train,initseed,expopt_all[1],ds),'rb') as fid:
                    accu2 = pickle.load(fid)
                for d in range(len(ds)):
                    accu_tmp = (accu1[d]+accu2[d])/2
                    accu_roi[d][m] = np.concatenate((accu_roi[d][m],np.mean(accu_tmp)[None]))
        idx = 0
        for d in range(len(ds)):
            for m in range(len(nfeature)):
                accu_roi_mean[idx] = np.mean(accu_roi[d][m])
                accu_roi_se[idx] = stats.sem(accu_roi[d][m])
                idx += 1
        op = out_path.format(model)
        if not os.path.exists(op):
            os.makedirs(op)     
        np.savez_compressed(op+'ds'+str(ds).replace(' ','')+'_'+roi+'.npz', feat=nfeature,accu_mean=accu_roi_mean,accu_se=accu_roi_se)
        print (op+roi)
        with open(op+'result_'+roi+'.txt','a') as fid:
            fid.write('nfeature: '+str(nfeature)+'\n')
            for d in ds:
                fid.write('dataset: '+str(d)+'\n')
                fid.write(str(accu_roi_mean[d*len(nfeature):(d+1)*len(nfeature)])+'\n')

