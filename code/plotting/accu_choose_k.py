#!/usr/bin/env python
import numpy as np
import yaml
import os
from scipy import stats
import pickle

setting = open('../setting.yaml')
options = yaml.safe_load(setting)

# # mysseg
# in_path = options['output_path']+'accu/mysseg/{}/{}_feat{}_ntrain{}_rand{}_{}_ds{}.pickle'
# out_path = options['output_path']+'accu_k/mysseg/{}/'

# nfeature = [25,50,75,100]
# num_train = 65
# ds = list(range(12))
# tst_ds = list(range(4))
# rand = [0, 2, 6]
# expopt_all = ['1st','2nd']
# model_all = ['multi_dict','indv_dict']
# roi_all = ['dmn','pt','eac']

# for model in model_all:
#     for roi in roi_all:
#         accu_roi = []
#         accu_roi_mean = np.zeros((len(tst_ds)*len(nfeature)),dtype=np.float32)
#         accu_roi_se = np.zeros((len(tst_ds)*len(nfeature)),dtype=np.float32)
#         for d in range(len(tst_ds)):
#             accu_roi.append([])
#             for m in range(len(nfeature)):
#                 accu_roi[d].append(np.empty((0),dtype=np.float32))
#         for m,feat in enumerate(nfeature):
#             for initseed in rand:
#                 try:
#                     with open(in_path.format(model,roi,feat,num_train,initseed,expopt_all[0],ds),'rb') as fid:
#                         accu1 = pickle.load(fid)
#                     with open(in_path.format(model,roi,feat,num_train,initseed,expopt_all[1],ds),'rb') as fid:
#                         accu2 = pickle.load(fid)
#                     for d in range(len(tst_ds)):
#                         accu_tmp = (accu1[d]+accu2[d])/2
#                         accu_roi[d][m] = np.concatenate((accu_roi[d][m],np.mean(accu_tmp)[None]))
#                 except:
#                     for d in range(len(tst_ds)):
#                         accu_roi[d][m] = 0.0
#                     print (model+' '+str(feat))
#         idx = 0
#         for d in range(len(tst_ds)):
#             for m in range(len(nfeature)):
#                 accu_roi_mean[idx] = np.mean(accu_roi[d][m])
#                 accu_roi_se[idx] = stats.sem(accu_roi[d][m])
#                 idx += 1
#         op = out_path.format(model)
#         if not os.path.exists(op):
#             os.makedirs(op)     
#         np.savez_compressed(op+'ds'+str(ds).replace(' ','')+'_'+roi+'.npz', feat=nfeature,accu_mean=accu_roi_mean,accu_se=accu_roi_se)
#         print (op+roi)
#         with open(op+'result_'+roi+'.txt','a') as fid:
#             fid.write('nfeature: '+str(nfeature)+'\n')
#             for d in tst_ds:
#                 fid.write('dataset: '+str(d)+'\n')
#                 fid.write(str(accu_roi_mean[d*len(nfeature):(d+1)*len(nfeature)])+'\n')



# # mapping, only use classification accuracy to choose k
# in_path = options['output_path']+'accu/mapping/{}/{}_feat{}_rand{}_{}_ds{}.npz'
# out_path = options['output_path']+'accu_k/mapping/{}/'

# nfeature = [25,50,75,100]
# num_train = 65
# ds = list(range(12))
# rand = [0, 2, 6]
# expopt_all = ['1st','2nd']
# # model_all = ['multi_srm','indv_srm','indv_ica','indv_gica']
# model_all = ['multi_dict','indv_dict']
# roi_all = ['dmn','pt','eac']

# for model in model_all:
#     for roi in roi_all:
#         accu_roi = np.zeros((len(rand),len(nfeature),4),dtype=np.float32)
#         for m,feat in enumerate(nfeature):
#             try:
#                 for r,initseed in enumerate(rand):
#                     ws = np.load(in_path.format(model,roi,feat,initseed,expopt_all[0],ds))
#                     accu1 = ws['accu_class']
#                     ws = np.load(in_path.format(model,roi,feat,initseed,expopt_all[1],ds))
#                     accu2 = ws['accu_class']                    
#                     accu_roi[r,m,:] = (accu1+accu2)/2
#             except:
#                 accu_roi[:,m,:] = 0.0
#                 print (model+' '+str(feat))
#         accu_mean = np.mean(accu_roi,axis=0)
#         accu_se = stats.sem(accu_roi,axis=0)
#         op = out_path.format(model)
#         if not os.path.exists(op):
#             os.makedirs(op)     
#         np.savez_compressed(op+'ds'+str(ds).replace(' ','')+'_'+roi+'.npz', feat=nfeature,accu_mean=accu_mean,accu_se=accu_se)
#         print (op+roi)
#         with open(op+'result_'+roi+'.txt','a') as fid:
#             fid.write('nfeature: '+str(nfeature)+'\n')
#             for d in range(4):
#                 fid.write('ds'+str(d)+'\n')
#                 fid.write(str(accu_mean[:,d])+'\n')


# imgpred, only use pmc
in_path = options['output_path']+'accu/imgpred/{}/{}_feat{}_rand{}_ds{}.npz'
out_path = options['output_path']+'accu_k/imgpred/{}/'

# num_train = 40
# ds = [0,1,2,3]
# rand = [0,1,2]

num_train = 65
ds = list(range(12))
rand = [0,1,2]

nfeature = [25,50,75,100]
model_all = ['multi_dict','indv_dict']
roi_all = ['pmc']

for model in model_all:
    for roi in roi_all:
        accu_roi = np.zeros((len(rand),len(nfeature)),dtype=np.float32)
        for m,feat in enumerate(nfeature):
            try:
                for r,initseed in enumerate(rand):                
                    ws = np.load(in_path.format(model,roi,feat,initseed,ds))
                    accu_roi[r,m] = np.mean(ws['accu'])
            except:
                accu_roi[:,m] = 0.0
                print (model+' '+str(feat))
        accu_mean = np.mean(accu_roi,axis=0)
        accu_se = stats.sem(accu_roi,axis=0)
        op = out_path.format(model)
        if not os.path.exists(op):
            os.makedirs(op)     
        np.savez_compressed(op+'ds'+str(ds).replace(' ','')+'_'+roi+'.npz', feat=nfeature,accu_mean=accu_mean,accu_se=accu_se)
        print (op+roi)
        with open(op+'result_'+roi+'.txt','a') as fid:
            fid.write('nfeature: '+str(nfeature)+'\n')
            fid.write(str(accu_mean)+'\n')

