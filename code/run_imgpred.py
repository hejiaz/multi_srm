#!/usr/bin/env python
# Run time segment matching experiment using multi-dataset SRM
# Alignment data is half of the movie/audiobook data of all datasets, prediction data is the other half
# By Hejia Zhang @ Princeton

import sys, os
sys.path.insert(0, os.path.abspath('..'))
import numpy as np
import yaml
import copy
import pickle
import importlib
import utils as ut
import random

# ds is the training datasets, always use sherlock (3) to test. Keep all shared subjects in sherlock, and use part of 
# its distinct subjects
# should only use pmc data
def run_expt(nfeature,initseed,loo_flag,model,roi,ds,num_loo=3):
	# parameters
	expt = 'imgpred'
	if model in ['multi_dict','indv_dict']:
		niter = 30
	else:
		niter = 50

	print (model)
	print (roi)

	# import alignment and experiment method
	if model in ['all_srm','indv_srm']:
		align = importlib.import_module('model.srm')
	elif model in ['all_ica','indv_ica']:
		align = importlib.import_module('model.ica')
	elif model in ['all_gica','indv_gica']:
		align = importlib.import_module('model.gica')
	elif model in ['multi_dict','indv_dict']:
		align = importlib.import_module('model.dictlearn')		
	elif model in ['multi_srm']:
		align = importlib.import_module('model.'+model)
	elif model in ['avg']:
		align = None
	else:
		raise Exception('not valid model')
		
	pred = importlib.import_module('experiment.'+expt)

	# load path
	try:
		setting = open('setting.yaml')
	except:
		setting = open('../setting.yaml')
	options = yaml.safe_load(setting)

	# load membership info
	ws = np.load(options['input_path']+'multi_srm/membership.npz')
	membership = ws['membership']

	# extract datasets we want to use
	tst_ds = ds.index(3) # the dataset index of sherlock
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
		# find sherlock's distinct subjects
		trn_ds = copy.copy(ds)
		trn_ds.remove(tst_ds)
		dist = []
		for m in range(nsubjs):
			if membership[m,tst_ds] != -1:
				if all(n==-1 for n in list(membership[m,trn_ds])):
					dist.append(m)
		num_dist = len(dist)
		new_order = list(range(num_dist))
		random.seed(initseed)
		random.shuffle(new_order)
		loo = []
		for m in range(num_loo):
			loo.append(dist[new_order[m]])
		trn_subj = [n for n in list(range(nsubjs)) if n not in loo]
		train_mb = membership[trn_subj,:]
		test_mb = membership[loo,:]

	# load input data
	print ('load data')
	# load movie data
	with open(options['input_path']+'multi_srm/{}_data_all.pickle'.format(roi),'rb') as fid:
	    data_tmp = pickle.load(fid)
	# load label
	with open(options['input_path']+'multi_srm/label.pickle','rb') as fid:
		label = pickle.load(fid)
	# load recall data (pmc)
	with open(options['input_path']+'multi_srm/recall_data_all.pickle','rb') as fid:
		recall_tmp = pickle.load(fid)

	# extract datasets we want to use and zscore
	data_align = []
	for d in range(len(ds)):
		data_align.append(ut.zscore_data_all(data_tmp[ds[d]]))
	del data_tmp
	data_pred = []
	for m in range(len(recall_tmp)):
		data_pred.append(ut.zscore_data(recall_tmp[m]))
	del recall_tmp

	# load location information for dictionary learning
	if model in ['multi_dict','indv_dict']:
		ws = np.load(options['input_path']+'multi_srm/roi_location.npz')
		loc = ws[roi]
		del ws

	print ('alignment')
	if model not in ['avg']:
		# alignment
		# S is the transformed alignment data from training subjects
		if model in ['multi_srm']:
			W,S,noise = align.align(data_align,train_mb,niter,nfeature,initseed,model)
		elif model in ['multi_dict','indv_dict']:
			W_grp,W,S= align.align(data_align,train_mb,niter,nfeature,initseed,model,loc)
		else:
			W,S = align.align(data_align,train_mb,niter,nfeature,initseed,model)
			# print (noise)
		# learn W_all, loo, and transform prediction data into shared space
		if model in ['multi_dict','indv_dict']:
			W_tst,loo_ds = ut.learn_W(data_align,S,W,train_mb,test_mb,tst_ds,model,W_grp)
		else:
			W_tst,loo_ds = ut.learn_W(data_align,S,W,train_mb,test_mb,tst_ds,model)
		transformed_pred = ut.transform_list(data_pred,W_tst,model)
		del S
	else:
		# average alignment data of training subjects as transformed alignment data
		_,loo_ds = ut.find_train_test_subj(train_mb,test_mb,tst_ds)
		transformed_pred = data_pred

	print ('run experiment')
	if loo_flag:
		accu = pred.predict_loo(transformed_pred,label,loo_ds)
	else:
		accu = pred.predict(transformed_pred,label)
	print ('accu: '+str(np.mean(accu)))

	# save results
	if not os.path.exists(options['output_path']+'accu/imgpred/'+model+'/'):
		os.makedirs(options['output_path']+'accu/imgpred/'+model+'/')
	out_file = options['output_path']+'accu/imgpred/'+model+'/'+'{}_feat{}_rand{}_ds{}.npz'.format(roi,nfeature,initseed,ds)
	np.savez_compressed(out_file,accu=accu)

