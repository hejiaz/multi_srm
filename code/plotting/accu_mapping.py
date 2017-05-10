#!/usr/bin/env python
import numpy as np
import yaml
import os
from scipy import stats
import pickle

setting = open('../setting.yaml')
options = yaml.safe_load(setting)

feat_dict = {'multi_srm':[75,150,75],'multi_dict':[75,50,50],'indv_srm':[75,50,75],'all_srm':[75,50,75],'indv_ica':[50,50,25],\
 'indv_gica':[50,50,50],'indv_dict':[50,25,50],'avg':[50,50,50]}
roi_all = ['dmn','pt','eac']
model_all = ['all_srm']

# mapping (average across subjects in each dataset)
in_path = options['output_path']+'accu/mapping/{}/{}_feat{}_rand{}_{}_ds{}.npz'
out_path = options['output_path']+'accu_bar/mapping/{}/{}/' # {} is total_ds,roi

rand = [0,2,6,7,8]
total_ds = list(range(12))

# rand = [0,1,3,5,6]
# total_ds = list(range(4))

ds = list(range(4))
expopt_all = ['1st','2nd']

for m,roi in enumerate(roi_all):
    op = out_path.format(total_ds,roi).replace(' ','')
    if not os.path.exists(op):
        os.makedirs(op) 
    for model in model_all:
        nfeat = feat_dict[model]
        accu_class_all = np.zeros((len(ds),len(rand)),dtype=np.float32)
        accu_rank_all = np.zeros((len(ds),len(rand)),dtype=np.float32)
        for r,initseed in enumerate(rand):
            try:
                for expopt in expopt_all:
                    ws = np.load(in_path.format(model,roi,nfeat[m],initseed,expopt,total_ds))
                    accu_class_all[:,r] += ws['accu_class']/2
                    accu_rank_all[:,r] += ws['accu_rank']/2
            except:
                print (model+'_'+roi+str(initseed))

        class_mean = np.mean(accu_class_all,axis=1)
        class_se = stats.sem(accu_class_all,axis=1)
        rank_mean = np.mean(accu_rank_all,axis=1)
        rank_se = stats.sem(accu_rank_all,axis=1)

        for d in range((len(ds))):   
            np.savez_compressed(op+model+'_ds'+str(ds[d])+'.npz',class_mean=class_mean[d],class_se=class_se[d],rank_mean=rank_mean[d],rank_se=rank_se[d])
    print (op)
	

# # mapping_all (average across subjects in each dataset)
# in_path = options['output_path']+'accu/mapping_all/{}/{}_feat{}_rand{}_{}_ds{}_tst{}.npz'
# out_path = options['output_path']+'accu_bar/mapping_all/{}/' # {} is roi

# tst_ds = [3]
# ds = [0,1,2,3]
# rand = [0,1,3,5,6]
# expopt_all = ['1st','2nd']

# for m,roi in enumerate(roi_all):
#     op = out_path.format(roi)
#     if not os.path.exists(op):
#         os.makedirs(op) 
#     for model in model_all:
#         nfeat = feat_dict[model]
#         for tst in tst_ds:
#             accu_class_all = np.zeros((len(rand)),dtype=np.float32)
#             accu_rank_all = np.zeros((len(rand)),dtype=np.float32)
#             for r,initseed in enumerate(rand):
#                 for expopt in expopt_all:
#                     ws = np.load(in_path.format(model,roi,nfeat[m],initseed,expopt,ds,tst))
#                     accu_class_all[r] += ws['accu_class'][0]/2
#                     accu_rank_all[r] += ws['accu_rank'][0]/2

#             class_mean = np.mean(accu_class_all)
#             class_se = stats.sem(accu_class_all)
#             rank_mean = np.mean(accu_rank_all)
#             rank_se = stats.sem(accu_rank_all)
 
#             np.savez_compressed(op+model+'_ds'+str(tst)+'.npz',class_mean=class_mean,class_se=class_se,rank_mean=rank_mean,rank_se=rank_se)
#     print (op)

# # mapping_loo (accuracy per subject in the left-out dataset (all subjects in that dataset, not just shared ones))
# in_path = options['output_path']+'accu/mapping_loo/{}/{}_feat{}_rand{}_loods{}.npz'
# out_path = options['output_path']+'accu_bar/mapping_loo/{}/' # {} is roi

# loo_ds_all = [1,2,3]
# rand = list(range(5))

# for m,roi in enumerate(roi_all):
#     op = out_path.format(roi)
#     if not os.path.exists(op):
#         os.makedirs(op) 
#     for model in model_all[:-1]: # all models except 'avg'
#         nfeat = feat_dict[model]
#         accu_class_all = []
#         accu_rank_all = []
#         # for each dataset, average across different random seeds
#         for loo_ds in loo_ds_all:
#             ws = np.load(in_path.format(model,roi,nfeat[m],rand[0],loo_ds))
#             accu_class = ws['accu_class']/len(rand)
#             accu_rank = ws['accu_rank']/len(rand)
#             for initseed in rand[1:]:            
#                 ws = np.load(in_path.format(model,roi,nfeat[m],initseed,loo_ds))
#                 accu_class += ws['accu_class']/len(rand)
#                 accu_rank += ws['accu_rank']/len(rand)      
#             accu_class_all.append(accu_class)          
#             accu_rank_all.append(accu_rank)  
#         # calculate mean and se
#         for d,loo_ds in enumerate(loo_ds_all):
#             class_mean = np.mean(accu_class_all[d])
#             class_se = stats.sem(accu_class_all[d])
#             rank_mean = np.mean(accu_rank_all[d])
#             rank_se = stats.sem(accu_rank_all[d])    
#             np.savez_compressed(op+model+'_ds'+str(loo_ds)+'.npz',class_mean=class_mean,class_se=class_se,rank_mean=rank_mean,rank_se=rank_se)

#     for model in [model_all[-1]]: # 'avg'
#         nfeat = feat_dict[model]
#         accu_class_all = []
#         accu_rank_all = []
#         # for each dataset, average across different random seeds
#         for loo_ds in loo_ds_all:
#             ws = np.load(in_path.format(model,roi,nfeat[m],0,loo_ds))    
#             accu_class_all.append(ws['accu_class'])          
#             accu_rank_all.append(ws['accu_rank'])  
#         # calculate mean and se
#         for d,loo_ds in enumerate(loo_ds_all):
#             class_mean = np.mean(accu_class_all[d])
#             class_se = stats.sem(accu_class_all[d])
#             rank_mean = np.mean(accu_rank_all[d])
#             rank_se = stats.sem(accu_rank_all[d])    
#             np.savez_compressed(op+model+'_ds'+str(loo_ds)+'.npz',class_mean=class_mean,class_se=class_se,rank_mean=rank_mean,rank_se=rank_se)       
#     print (op)



