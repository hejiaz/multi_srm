#!/usr/bin/env python
# Run fMRI to word embedding experiment using multi-dataset SRM. Leave out a whole dataset (loo_ds) for testing.
# Learn a global linear mapping for data from the other datasets. Use the shared subjects between loo_ds
# and the other datasets to learn W for loo_ds
# By Hejia Zhang @ Princeton

import numpy as np
import yaml
import os
import pickle
import importlib
import utils as ut
import random
import sys, os
sys.path.insert(0, os.path.abspath('..'))


# loo_ds = 1 # which datasets to leave out: greeneye,milky,vodka,sherlock

def run_expt(nfeature,initseed,model,roi,loo_ds):
	# parameters
	expt = 'mapping_loo'
	train_ds = [0,1,2,3]
	word_ds = [0,1,2,3]
	niter = 50
	num_previous = 8
	word_dim = 300
	num_chunks = [7,4,4,25] # different number of scenes for different datasets, make sure the scene length is not too short
	pre = ''

	print (model)
	print (roi)

	# import alignment and experiment method
	if model in ['all_srm','indv_srm']:
		align = importlib.import_module('model.srm')
	elif model in ['all_ica','indv_ica']:
		align = importlib.import_module('model.ica')
	elif model in ['all_gica','indv_gica']:
		align = importlib.import_module('model.gica')
	elif model in ['all_dict','indv_dict']:
		align = importlib.import_module('model.dictlearn')		
	elif model in ['multi_srm']:
		align = importlib.import_module('model.'+model)
	elif model in ['avg']:
		align = None
	else:
		raise Exception('not valid model')
		
	pred = importlib.import_module('experiment.'+expt)

	# load path
	# setting = open('setting.yaml')
	setting = open('../setting.yaml')
	options = yaml.safe_load(setting)

	# load membership info
	ws = np.load(options['input_path']+'multi_srm/membership.npz')
	membership = ws['membership']

	# load location information for dictionary learning
	if model in ['all_dict','indv_dict']:
		ws = np.load(options['input_path']+'multi_srm/roi_location.npz')
		loc = ws[roi]
		del ws

	# separate training membership 
	train_ds.remove(loo_ds)
	train_mb = membership[:,train_ds]
	train_mb = ut.remove_invalid_membership(train_mb)
	# check if the training datasets can be linked through shared subjects
	if not ut.check_membership(train_mb):
		raise Exception('Not all datasets can be linked through shared training subjects')

	# find shared subjects between train_ds and loo_ds and their indices
	idx_loo,idx_train = ut.find_shared_subjects_loo_ds(membership,loo_ds)

	# load input data
	print ('load data')
	with open(options['input_path']+'multi_srm/{}_data_all.pickle'.format(roi),'rb') as fid:
	    data = pickle.load(fid)
	with open(options['input_path']+'multi_srm/'+pre+'word_embedding{}_all.pickle'.format(word_dim),'rb') as fid:
	    word = pickle.load(fid)



	# extract datasets we want to use
	data_align = []
	word_align = []
	for d in range(len(train_ds)):
		data_align.append(data[train_ds[d]])
		word_align.append(word[train_ds[d]])

	data_pred = data[loo_ds]
	word_pred = word[loo_ds]

	new_ds = []
	for w in word_ds:
		try:
			new_ds.append(train_ds.index(w))
		except:
			continue

	# extract data of shared subjects in left-out dataset for learning S_loo later
	data_loo_shared = data_pred[:,:,idx_loo]

	print ('alignment')
	if model not in ['avg']:
		# alignment
		# S is the transformed alignment data from training subjects
		if model in ['multi_srm']:
			W,S,noise = align.align(data_align,train_mb,niter,nfeature,initseed,model)
		elif model in ['all_dict','indv_dict']:
			W_grp,W,S= align.align(data_align,train_mb,niter,nfeature,initseed,model,loc)
		else:
			W,S = align.align(data_align,train_mb,niter,nfeature,initseed,model)
	else:
		# average alignment data of training subjects as transformed alignment data
		S = []
		for d in range(len(train_ds)):
			S.append(np.mean(data_align[d],axis=2))

	print ('learn linear mapping')
	# learn linear mapping using:
	# 1) concatenate transformed alignment data from training datasets
	# 2) concatenate word embeddings for alignment data
	word_align_all,word_pred_all = pred.process_semantic([word_align[n] for n in new_ds],word_pred,num_previous)
	W_ft = pred.learn_linear_map([S[n] for n in new_ds],word_align_all,num_previous)

	print ('transform left-out dataset')
	if model not in ['avg']:
		# learn S using shared subjects
		S_loo = np.mean(ut.transform(data_loo_shared,W[:,:,idx_train],model),axis=2)
		# learn W of all subjects using S_loo
		W_loo = ut.learn_W_loo_ds(data_pred,S_loo,model)
		transformed_pred = ut.transform(data_pred,W_loo,model)
	else:
		transformed_pred = data_pred	

	print ('run experiment')
	accu_class,accu_rank = pred.predict(transformed_pred,word_pred_all,W_ft,num_chunks[loo_ds],num_previous)

	print ('results:')
	print ('accu_class: '+str(np.mean(accu_class)))
	print ('accu_rank: '+str(np.mean(accu_rank)))

	# save results
	if not os.path.exists(options['output_path']+'accu/mapping_loo/'+model+'/'):
		os.makedirs(options['output_path']+'accu/mapping_loo/'+model+'/')
	np.savez_compressed(options['output_path']+'accu/mapping_loo/'+model+'/{}_feat{}_rand{}_loods{}.npz'.format(roi,nfeature,initseed,loo_ds),accu_class=accu_class,accu_rank=accu_rank)



