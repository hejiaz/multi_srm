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
import sys, os
sys.path.insert(0, os.path.abspath('..'))


# portion should be (0,1), how much portion of secondary datasets to use in terms of number of TRs
def run_expt(nfeature,initseed,expopt,num_train,loo_flag,model,roi,ds,portion):
	# parameters
	expt = 'mysseg'
	niter = 50

	model = 'multi_srm'
	print (model)
	print (roi)

	# import alignment and experiment method
	align = importlib.import_module('model.'+model)		
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

	# TR information for all datasets
	TR_1st = []
	TR_used = []
	for d in range(ndata):
		TR_1st.append(int(data[d].shape[1]/2))
		TR_used.append(int((data[d].shape[1])*portion))

	# different alignment and prediction data for different datasets
	accu = []
	for pd in range(4):
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
				data_align.append(ut.zscore_data_all(data[d][:,:TR_used[d],:]))
				data_pred.append(None)

		# alignment
		# S is the transformed alignment data from training subjects
		if model in ['multi_srm']:
			W,S,noise = align.align(data_align,train_mb,niter,nfeature,initseed,model)
		elif model in ['multi_dict','all_dict','indv_dict']:
			W_grp,W,S= align.align(data_align,train_mb,niter,nfeature,initseed,model,loc)
		else:
			W,S = align.align(data_align,train_mb,niter,nfeature,initseed,model)
		# learn W_all, loo, and transform prediction data into shared space
		transformed_pred = []
		loo = []
		for d in range(4):
			if d == pd:
				W_tmp,loo_tmp = ut.learn_W(data_align,S,W,train_mb,test_mb,d,model)
				transformed_pred.append(ut.transform(data_pred[d],W_tmp,model))
				loo.append(loo_tmp)
			else:
				transformed_pred.append(None)
				loo.append(None)
		del S

		# experiment
		if loo_flag:
			accu.append(pred.predict_loo(transformed_pred[pd],loo[pd]))
		else:
			accu.append(pred.predict(transformed_pred[pd]))
		print ('accu'+str(pd)+' : '+str(np.mean(accu[pd])))

	# save results
	if not os.path.exists(options['output_path']+'accu/different_TR/'+model+'/'):
		os.makedirs(options['output_path']+'accu/different_TR/'+model+'/')
	out_file = options['output_path']+'accu/different_TR/'+model+'/'+'{}_feat{}_ntrain{}_rand{}_{}_ds{}_{}portion.pickle'.format(roi,nfeature,num_train,initseed,expopt,ds,portion)
	with open(out_file,'wb') as fid:
		pickle.dump(accu,fid,pickle.HIGHEST_PROTOCOL)

