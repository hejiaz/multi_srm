#!/usr/bin/env python
import numpy as np
import yaml
import os
from scipy import stats
import pickle

setting = open('../setting.yaml')
options = yaml.safe_load(setting)

feat_dict = {'multi_srm':[125],'multi_dict':[100],'indv_srm':[125],'all_srm':[125],'indv_dict':[100],'indv_ica':[25],'indv_gica':[25],'avg':[50]}

ds = list(range(12))
rand = list(range(5))

roi_all = ['pmc']

# imgpred
in_path = options['output_path']+'accu/imgpred/{}/{}_feat{}_rand{}_ds{}.npz'
out_path = options['output_path']+'accu_bar/imgpred/{}/{}/' # {} is ds,roi
# model_all = ['multi_srm','indv_srm','multi_dict','indv_dict','indv_ica','indv_gica','avg']
model_all = ['indv_srm']
for m,roi in enumerate(roi_all):
    op = out_path.format(ds,roi).replace(' ','')
    if not os.path.exists(op):
        os.makedirs(op)     
    for model in model_all:
        nfeat = feat_dict[model]
        accu_all = np.zeros((len(rand)),dtype=np.float32)
        try:
            for r,initseed in enumerate(rand):
                ws = np.load(in_path.format(model,roi,nfeat[m],initseed,ds))
                accu_all[r] = np.mean(ws['accu'])
        except:
            accu_all = 0.
            print (model+'_'+roi)
        accu_mean = np.mean(accu_all)
        accu_se = stats.sem(accu_all)
        np.savez_compressed(op+model+'.npz', mean=accu_mean,se=accu_se)
        print (op+model)

