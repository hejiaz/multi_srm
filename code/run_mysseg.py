#!/usr/bin/env python
# Run time segment matching experiment using multi-dataset SRM
# Alignment data is half of the movie/audiobook data of all datasets, prediction data is the other half
# By Hejia Zhang @ Princeton

import numpy as np
import yaml
import os,copy
import pickle
import importlib
import utils as ut
import random
import sys, os
sys.path.insert(0, os.path.abspath('..'))

def run_expt(nfeature,initseed,expopt,num_train,loo_flag,model,roi,ds):
	# parameters
	expt = 'mysseg'
	if model in ['multi_dict','indv_dict']:
		niter = 30
	else:
		niter = 50

	tst_ds = [0,1,2,3]
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

	# separate alignment and prediction data
	TR_1st = []
	for d in range(len(tst_ds)):
		TR_1st.append(int(data[d].shape[1]/2))
	data_1st = []
	data_2nd = []
	for d in range(ndata):
		if d in tst_ds:
			data_1st.append(ut.zscore_data_all(data[d][:,:TR_1st[d],:]))
			data_2nd.append(ut.zscore_data_all(data[d][:,TR_1st[d]:,:]))
		else:
			data_1st.append(ut.zscore_data_all(data[d]))
			data_2nd.append(ut.zscore_data_all(data[d]))
	if expopt == '1st':
		data_align = data_2nd
		data_pred = data_1st
	elif expopt == '2nd':
		data_align = data_1st
		data_pred = data_2nd
	else:
		raise Exception('expopt has to be 1st or 2nd')
	del data

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
		# W_all = []
		transformed_pred = []
		loo = []
		for d in tst_ds:
			if model in ['multi_dict','indv_dict']:
				W_tmp,loo_tmp = ut.learn_W(data_align,S,W,train_mb,test_mb,d,model,W_grp)
			else:
				W_tmp,loo_tmp = ut.learn_W(data_align,S,W,train_mb,test_mb,d,model)
			transformed_pred.append(ut.transform(data_pred[d],W_tmp,model))
			# W_all.append(W_tmp)
			loo.append(loo_tmp)
		# # save W
		# if not os.path.exists(options['output_path']+'W/mysseg/'+model+'/'):
		# 	os.makedirs(options['output_path']+'W/mysseg/'+model+'/')
		# with open(options['output_path']+'W/mysseg/'+model+'/'+'{}_feat{}_ntrain{}_rand{}_{}_ds{}.pickle'.format(roi,nfeature,num_train,initseed,expopt,ds),'wb') as fid:
		# 	pickle.dump(W_all,fid,pickle.HIGHEST_PROTOCOL)
		# del W_all
		del S
	else:
		# average alignment data of training subjects as transformed alignment data
		loo = []
		for d in tst_ds:
			_,test_subj = ut.find_train_test_subj(train_mb,test_mb,d)
			loo.append(test_subj)
		transformed_pred = data_pred

	print ('run experiment')
	accu = []
	for d in tst_ds:
		if loo_flag:
			accu.append(pred.predict_loo(transformed_pred[d],loo[d]))
		else:
			accu.append(pred.predict(transformed_pred[d]))
		print ('accu'+str(d)+' : '+str(np.mean(accu[d])))

	# save results
	if not os.path.exists(options['output_path']+'accu/mysseg/'+model+'/'):
		os.makedirs(options['output_path']+'accu/mysseg/'+model+'/')
	out_file = options['output_path']+'accu/mysseg/'+model+'/'+'{}_feat{}_ntrain{}_rand{}_{}_ds{}.pickle'.format(roi,nfeature,num_train,initseed,expopt,ds)
	with open(out_file,'wb') as fid:
		pickle.dump(accu,fid,pickle.HIGHEST_PROTOCOL)

