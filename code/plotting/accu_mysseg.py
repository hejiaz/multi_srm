#!/usr/bin/env python
import numpy as np
import yaml
import os
from scipy import stats
import pickle

setting = open('../setting.yaml')
options = yaml.safe_load(setting)

feat_dict = {'multi_srm':[75,75,100],'multi_dict':[25,50,50],'indv_srm':[75,75,50],'all_srm':[75,75,50],'indv_ica':[50,25,25],\
'indv_gica':[50,50,25],'indv_dict':[50,25,50],'avg':[50,50,50]}

tst_ds = list(range(4))

# num_train = 65
# ds = list(range(12))
# rand = [0,2,6,7,8]

num_train = 40
ds = list(range(4))
rand = [0,1,3,5,6]

expopt_all = ['1st','2nd']
roi_all = ['dmn','pt','eac']

# # mysseg
# in_path = options['output_path']+'accu/mysseg/{}/{}_feat{}_ntrain{}_rand{}_{}_ds{}.pickle'
# out_path = options['output_path']+'accu_bar/mysseg/{}/{}/' # {} is ds,roi
# model_all = ['all_srm']
# for m,roi in enumerate(roi_all):
#     op = out_path.format(ds,roi).replace(' ','')
#     if not os.path.exists(op):
#         os.makedirs(op)     
#     for model in model_all:
#         nfeat = feat_dict[model]
#         accu_all = np.zeros((len(tst_ds),len(rand)),dtype=np.float32)
#         for r,initseed in enumerate(rand):
#             try:
#                 with open(in_path.format(model,roi,nfeat[m],num_train,initseed,expopt_all[0],ds),'rb') as fid:
#                     accu1 = pickle.load(fid)
#                 with open(in_path.format(model,roi,nfeat[m],num_train,initseed,expopt_all[1],ds),'rb') as fid:
#                     accu2 = pickle.load(fid)
#                 for d in range(len(tst_ds)):
#                     accu_tmp = (accu1[d]+accu2[d])/2
#                     accu_all[d,r] = np.mean(accu_tmp)
#             except:
#                 accu_all[:,r] = 0.
#                 print (model+'_'+roi+str(initseed))
#         accu_mean = np.mean(accu_all,axis=1)
#         accu_se = stats.sem(accu_all,axis=1)
#         for d in range(len(tst_ds)):
#             np.savez_compressed(op+model+'_ds'+str(tst_ds[d])+'.npz', mean=accu_mean[d],se=accu_se[d])
#     print (op)


# mysseg_all
in_path = options['output_path']+'accu/mysseg_all/{}/{}_feat{}_ntrain{}_rand{}_{}_ds{}.pickle'
if num_train == 40:
    out_path = options['output_path']+'accu_bar/mysseg_all_old/{}/' # {} is roi
else:
    out_path = options['output_path']+'accu_bar/mysseg_all/{}/' # {} is roi
model_all = ['multi_srm']
for m,roi in enumerate(roi_all):
    op = out_path.format(roi)
    if not os.path.exists(op):
        os.makedirs(op)     
    for model in model_all:
        nfeat = feat_dict[model]
        accu_all = np.zeros((len(tst_ds),len(rand)),dtype=np.float32)
        for r,initseed in enumerate(rand):
            with open(in_path.format(model,roi,nfeat[m],num_train,initseed,expopt_all[0],ds),'rb') as fid:
                accu1 = pickle.load(fid)
            with open(in_path.format(model,roi,nfeat[m],num_train,initseed,expopt_all[1],ds),'rb') as fid:
                accu2 = pickle.load(fid)
            for d in range(len(tst_ds)):
                accu_tmp = (accu1[d]+accu2[d])/2
                accu_all[d,r] = np.mean(accu_tmp)
        accu_mean = np.mean(accu_all,axis=1)
        accu_se = stats.sem(accu_all,axis=1)
        for d in range(len(tst_ds)):
            np.savez_compressed(op+model+'_ds'+str(tst_ds[d])+'.npz', mean=accu_mean[d],se=accu_se[d])
    print (op)


