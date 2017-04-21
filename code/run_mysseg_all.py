#!/usr/bin/env python
# Run time segment matching experiment on multi-dataset SRM
# Can use all TRs from all datasets except for the prediction dataset
# By Hejia Zhang @ Princeton

import numpy as np
import yaml
import os
import pickle
import importlib
import utils as ut
import random

# parameters
expt = 'mysseg'
niter = 10
nfeature = 25
initseed = 0
expopt = '2nd'

num_train = 30
loo_flag = True

# model = ['multi_srm','srm_rotate','srm_rotate_ind','indv_srm','avg']
model = 'srm_rotate_ind'
roi = 'pt'
ds = [0,1,2,3] # which datasets to use: greeneye,milky,vodka,sherlock

print (model)
print (roi)

# import alignment and experiment method
if model in ['srm_rotate_ind']:
	align = importlib.import_module('model.srm_rotate')
elif model in ['multi_srm','srm_rotate','indv_srm']:
	align = importlib.import_module('model.'+model)
elif model in ['avg']:
	align = None
else:
	raise Exception('not valid model')
	
pred = importlib.import_module('experiment.'+expt)

# load path
setting = open('setting.yaml')
options = yaml.safe_load(setting)

# load membership info
ws = np.load(options['input_path']+'multi_srm/membership.npz')
membership = ws['membership']

# extract datasets we want to use
membership = membership[:,ds]
membership = ut.remove_invalid_membership(membership)
nsubjs,ndata = membership.shape

# separate training and testing subjects
if not loo_flag:
	# version 1: no left-out subjects
	train_mb = membership
	test_mb = np.empty((0,ndata),dtype=np.int32)
	num_train = nsubjs
else:
	# version 2: with left-out subjects
	new_order = list(range(nsubjs))
	random.seed(initseed)
	random.shuffle(new_order)
	membership = np.array([membership[n,:] for n in new_order])
	train_mb = membership[:num_train,:]
	test_mb = membership[num_train:,:]

# check if the training subjects can be linked through shared subjects
if not ut.check_membership(train_mb):
	raise Exception('Not all datasets can be linked through shared training subjects')
# check if all datasets have left-out testing subjects
if test_mb.shape[0]:
	if not ut.check_test_mb(test_mb):
		raise Exception('Not all datasets has testing subjects')

# load input data
print ('load data')
with open(options['input_path']+'multi_srm/{}_data_all.pickle'.format(roi),'rb') as fid:
    data_tmp = pickle.load(fid)

# extract datasets we want to use
data = []
for d in range(len(ds)):
	data.append(data_tmp[ds[d]])
del data_tmp

# TR information for all datasets
TR_1st = []
for d in range(ndata):
	TR_1st.append(int(data[d].shape[1]/2))

# different alignment and prediction data for different datasets
accu = []
for pd in range(ndata):
	print ('dataset'+str(pd))
	# separate alignment and prediction data
	data_align = []
	data_pred = []
	for d in range(ndata):
		if d == pd:
			if expopt == '1st':
				data_align.append(ut.zscore_data_all(data[d][:,TR_1st[d]:,:]))
				data_pred.append(ut.zscore_data_all(data[d][:,:TR_1st[d],:]))
			elif expopt == '2nd':
				data_align.append(ut.zscore_data_all(data[d][:,:TR_1st[d],:]))
				data_pred.append(ut.zscore_data_all(data[d][:,TR_1st[d]:,:]))
			else:
				raise Exception('expopt has to be 1st or 2nd')
		else:
			data_align.append(ut.zscore_data_all(data[d]))
			data_pred.append(None)
	# alignment
	if model not in ['avg']:
		# alignment
		# S is the transformed alignment data from training subjects
		W,S = align.align(data_align,train_mb,niter,nfeature,initseed)
		# learn W_all, loo, and transform prediction data into shared space
		W_all = []
		transformed_pred = []
		loo = []
		for d in range(ndata):
			if d == pd:
				W_tmp,loo_tmp = ut.learn_W(data_align,S,W,train_mb,test_mb,d,model)
				transformed_pred.append(ut.transform(data_pred[d],W_tmp))
				W_all.append(W_tmp)
				loo.append(loo_tmp)
			else:
				transformed_pred.append(None)
				W_all.append(None)
				loo.append(None)
		# # save W
		# if not os.path.exists(options['output_path']+'W/mysseg_all/'+model+'/'):
		# 	os.makedirs(options['output_path']+'W/mysseg_all/'+model+'/')
		# with open(options['output_path']+'W/mysseg_all/'+model+'/'+'{}_feat{}_ntrain{}_rand{}_{}_ds{}.pickle'.format(roi,nfeature,num_train,initseed,expopt,ds),'wb') as fid:
		# 	pickle.dump(W_all,fid,pickle.HIGHEST_PROTOCOL)
		del W_all
		del S
	else:
		# average alignment data of training subjects as transformed alignment data
		loo = []
		for d in range(ndata):
			if d == pd:
				_,test_subj = ut.find_train_test_subj(train_mb,test_mb,d)
				loo.append(test_subj)
			else:
				loo.append(None)
		transformed_pred = data_pred
	# experiment
	if loo_flag:
		accu.append(pred.predict_loo(transformed_pred[pd],loo[pd]))
	else:
		accu.append(pred.predict(transformed_pred[pd]))
	print ('accu'+str(pd)+' : '+str(np.mean(accu[pd])))


# save results
if not os.path.exists(options['output_path']+'accu/mysseg_all/'+model+'/'):
	os.makedirs(options['output_path']+'accu/mysseg_all/'+model+'/')
out_file = options['output_path']+'accu/mysseg_all/'+model+'/'+'{}_feat{}_ntrain{}_rand{}_{}_ds{}.pickle'.format(roi,nfeature,num_train,initseed,expopt,ds)
with open(out_file,'wb') as fid:
	pickle.dump(accu,fid,pickle.HIGHEST_PROTOCOL)

