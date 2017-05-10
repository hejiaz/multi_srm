#!/usr/bin/env python
import numpy as np
import yaml
import os
from scipy import stats
import pickle

setting = open('../setting.yaml')
options = yaml.safe_load(setting)

feat_dict = {'multi_srm':[75,75,100],'multi_dict':[25,50,50],'indv_srm':[75,75,50],'indv_ica':[50,25,25],\
'indv_gica':[50,50,25],'indv_dict':[50,25,50],'avg':[50,50,50]}
roi_all = ['dmn','pt','eac']

# # loo_ds
# in_path = options['output_path']+'accu/loo_ds/{}/{}_feat{}_rand{}_loo{}_other{}.pickle'
# out_path = options['output_path']+'accu_bar/loo_ds/{}/' # {} is roi
# model_all = ['multi_srm']
# loo_ds_all = [0,3,1]
# other_ds_all = [[1,3],[1,0],[0,3]]
# assert(len(loo_ds_all)==len(other_ds_all))
# rand = list(range(5))

# for m,roi in enumerate(roi_all):
# 	op = out_path.format(roi)
# 	if not os.path.exists(op):
# 		os.makedirs(op) 
# 	for model in model_all:
# 		nfeat = feat_dict[model]
# 		for loo,other in zip(loo_ds_all,other_ds_all):
# 			accu_tmp = [] 
# 			accu_mean = np.zeros((len(other)),dtype=np.float32)
# 			accu_se = np.zeros((len(other)),dtype=np.float32)
# 			# average accuracy across different random seeds
# 			with open(in_path.format(model,roi,nfeat[m],rand[0],loo,other),'rb') as fid:
# 				accu_rand = pickle.load(fid)
# 			for d in range(len(other)):
# 				accu_tmp.append(accu_rand[d]/len(rand))
# 			for initseed in rand[1:]:
# 				with open(in_path.format(model,roi,nfeat[m],initseed,loo,other),'rb') as fid:
# 					accu_rand = pickle.load(fid)
# 				for d in range(len(other)):
# 					accu_tmp[d] += accu_rand[d]/len(rand)
# 			# calculate mean and se of each other_ds
# 			for d in range(len(other)):
# 				num_shared = len(accu_tmp[d])
# 				accu_mean[d] = np.mean(accu_tmp[d])
# 				accu_se[d] = stats.sem(accu_tmp[d])
# 			# save results  
# 			np.savez_compressed(op+model+'_loo{}_other{}.npz'.format(loo,other).replace(' ',''), mean=accu_mean,se=accu_se,subj=num_shared)
# 	print (op)


# # overfit
# in_path = options['output_path']+'accu/overfit/{}/{}_feat{}_rand{}_{}_ds{}.npz'
# out_path = options['output_path']+'accu_bar/overfit/{}/' # {} is roi
# # model_all = ['indv_srm','indv_ica','indv_gica','indv_dict']
# model_all = ['indv_dict']
# ds_all = [[0,1],[0,2],[0,3],[1,3]]
# expopt_all = ['1st','2nd']
# rand = list(range(5))
# num_rand = len(expopt_all)*len(rand)

# for m,roi in enumerate(roi_all):
# 	op = out_path.format(roi)
# 	if not os.path.exists(op):
# 		os.makedirs(op) 
# 	for model in model_all:
# 		nfeat = feat_dict[model]
# 		for ds in ds_all:			
# 			# order: ds1 using own;ds1 using other;ds2 using own;ds2 using other
# 			accu_mean = np.zeros((4),dtype=np.float32)
# 			accu_se = np.zeros((4),dtype=np.float32)
# 			# average accuracy across different random seeds and expopt
# 			accu_tmp = []
# 			for e,expopt in enumerate(expopt_all):
# 				for r,initseed in enumerate(rand):
# 					accu_rand = np.load(in_path.format(model,roi,nfeat[m],initseed,expopt,ds).replace(' ',''))
# 					if e==0 and r==0:						
# 						accu_tmp.append(accu_rand['accu1'][:,0]/num_rand)
# 						accu_tmp.append(accu_rand['accu1'][:,1]/num_rand)
# 						accu_tmp.append(accu_rand['accu2'][:,0]/num_rand)
# 						accu_tmp.append(accu_rand['accu2'][:,1]/num_rand)
# 					else:
# 						accu_tmp[0] += accu_rand['accu1'][:,0]/num_rand
# 						accu_tmp[1] += accu_rand['accu1'][:,1]/num_rand
# 						accu_tmp[2] += accu_rand['accu2'][:,0]/num_rand
# 						accu_tmp[3] += accu_rand['accu2'][:,1]/num_rand
# 			# calculate mean and se of each column
# 			for d in range(4):
# 				accu_mean[d] = np.mean(accu_tmp[d])
# 				accu_se[d] = stats.sem(accu_tmp[d])
# 			# save results  
# 			np.savez_compressed(op+model+'_ds{}.npz'.format(ds).replace(' ',''), mean=accu_mean,se=accu_se)
# 	print (op)


# # shared/dist subjects
# # exp = 'shared_subj'
# exp = 'dist_subj'
# in_path = options['output_path']+'accu/'+exp+'/{}/{}_feat{}_rand{}_{}_ds{}_shared{}.npz'
# out_path = options['output_path']+'accu_bar/'+exp+'/{}/' # {} is roi
# roi_all = ['dmn']
# model_all = ['multi_srm']
# shared_all = [4]
# ds_all = [[0,3]]
# expopt_all = ['1st','2nd']
# rand = list(range(5))

# for m,roi in enumerate(roi_all):
# 	op = out_path.format(roi)
# 	if not os.path.exists(op):
# 		os.makedirs(op) 
# 	for model in model_all:
# 		nfeat = feat_dict[model]
# 		for ds in ds_all:
# 			for shared in shared_all:
# 				# read first rand (must have first rand)
# 				ws = np.load(in_path.format(model,roi,nfeat[m],rand[0],expopt_all[0],ds,shared))
# 				accu1 = ws['accu']
# 				ws = np.load(in_path.format(model,roi,nfeat[m],rand[0],expopt_all[1],ds,shared))
# 				accu2 = ws['accu']
# 				accu = ((accu1+accu2)/2)[:,None]
# 				# read other rands
# 				for initseed in rand[1:]:
# 					try:
# 						ws = np.load(in_path.format(model,roi,nfeat[m],initseed,expopt_all[0],ds,shared))
# 						accu1 = ws['accu']
# 						ws = np.load(in_path.format(model,roi,nfeat[m],initseed,expopt_all[1],ds,shared))
# 						accu2 = ws['accu']
# 						accu = np.concatenate((accu,((accu1+accu2)/2)[:,None]),axis=1)
# 					except:
# 						print (roi+'_'+str(ds)+'_rand'+str(initseed))
# 				accu_mean = np.mean(accu,axis=1)
# 				accu_se = stats.sem(accu,axis=1)
# 				# save results  
# 				np.savez_compressed(op+model+'_ds{}_shared{}.npz'.format(ds,shared).replace(' ',''), mean=accu_mean,se=accu_se)
# 	print (op)



# correlation
in_path = options['output_path']+'accu/corr/{}/{}_feat{}_rand{}_{}_ds{}.npz'
out_path = options['output_path']+'accu_bar/corr/{}/' # {} is roi
model_all = ['indv_srm','avg']
nfeat = 75
ds_all = [0,3]
expopt_all = ['1st','2nd']
rand = list(range(5))

for m,roi in enumerate(roi_all):
	op = out_path.format(roi)
	if not os.path.exists(op):
		os.makedirs(op) 
	for model in model_all:
		for ds in ds_all:
			sep = np.zeros((len(rand)),dtype=np.float32)
			tgr = np.zeros((len(rand)),dtype=np.float32)
			for r,initseed in enumerate(rand):			
				ws = np.load(in_path.format(model,roi,nfeat,initseed,expopt_all[0],ds))
				sep1 = ws['sep']
				tgr1 = ws['tgr']
				ws = np.load(in_path.format(model,roi,nfeat,initseed,expopt_all[1],ds))
				sep2 = ws['sep']
				tgr2 = ws['tgr']
				sep[r] = (sep1+sep2)/2
				tgr[r] = (tgr1+tgr2)/2
			sep_mean = np.mean(sep)
			sep_se = stats.sem(sep)
			tgr_mean = np.mean(tgr)
			tgr_se = stats.sem(tgr)
			# save results  
			np.savez_compressed(op+model+'_ds{}.npz'.format(ds), sep_mean=sep_mean,sep_se=sep_se,tgr_mean=tgr_mean,tgr_se=tgr_se)
	print (op)


