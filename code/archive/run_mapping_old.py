#!/usr/bin/env python
# Run fMRI to word embedding experiment using multi-dataset SRM. Learn a global linear mapping for data from 
# all datasets. Half of the movie and word embedding as alignment data, and the other half as testing
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

# use ds to train, use word_ds to test. Only word_ds has text embeddings

def run_expt(nfeature,initseed,expopt,num_train,loo_flag,model,roi,ds):
	# parameters
	expt = 'mapping'
	niter = 50
	num_previous = 8
	word_dim = 300
	word_ds = [0,1,2,3]
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
	elif model in ['multi_dict','all_dict','indv_dict']:
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

	# load location information for dictionary learning
	if model in ['multi_dict','all_dict','indv_dict']:
		ws = np.load(options['input_path']+'multi_srm/roi_location.npz')
		loc = ws[roi]
		del ws

	# load membership info
	ws = np.load(options['input_path']+'multi_srm/membership.npz')
	membership = ws['membership']

	# extract datasets we want to use
	membership = membership[:,ds]
	membership = ut.remove_invalid_membership(membership)
	nsubjs,ndata = membership.shape

	# find position of word_ds in ds
	new_ds = []
	for w in word_ds:
		new_ds.append(ds.index(w))

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
	with open(options['input_path']+'multi_srm/'+pre+'word_embedding{}_all.pickle'.format(word_dim),'rb') as fid:
	    word_tmp = pickle.load(fid)

	# extract datasets we want to use
	data = []
	word = []
	for d in range(len(ds)):
		data.append(data_tmp[ds[d]])
		word.append(word_tmp[ds[d]])
	del data_tmp
	del word_tmp

	# separate alignment and prediction data
	TR_1st = []
	for d in range(ndata):
		TR_1st.append(int(data[d].shape[1]/2))
	data_1st = []
	data_2nd = []
	word_1st = []
	word_2nd = []
	for d in range(ndata):
		data_1st.append(ut.zscore_data_all(data[d][:,:TR_1st[d],:]))
		data_2nd.append(ut.zscore_data_all(data[d][:,TR_1st[d]:,:]))
		if word[d] is None:
			word_1st.append([])
			word_2nd.append([])
		else:
			word_1st.append(word[d][:,:TR_1st[d]])
			word_2nd.append(word[d][:,TR_1st[d]:])
	if expopt == '1st':
		data_align = data_2nd
		data_pred = data_1st
		word_align = word_2nd
		word_pred = word_1st
	elif expopt == '2nd':
		data_align = data_1st
		data_pred = data_2nd
		word_align = word_1st
		word_pred = word_2nd
	else:
		raise Exception('expopt has to be 1st or 2nd')
	del data
	del word

	print ('alignment')
	if model not in ['avg']:
		# alignment
		# S is the transformed alignment data from training subjects
		if model in ['multi_srm']:
			W,S,noise = align.align(data_align,train_mb,niter,nfeature,initseed,model)
		elif model in ['multi_dict','all_dict','indv_dict']:
			W_grp,W,S= align.align(data_align,train_mb,niter,nfeature,initseed,model,loc)
		else:
			W,S = align.align(data_align,train_mb,niter,nfeature,initseed,model)
		# learn W_all, loo, and transform prediction data into shared space
		# W_all = []
		transformed_pred = []
		loo = []
		for d in range(ndata):
			if model in ['multi_dict','all_dict','indv_dict']:
				W_tmp,loo_tmp = ut.learn_W(data_align,S,W,train_mb,test_mb,d,model,W_grp)
			else:
				W_tmp,loo_tmp = ut.learn_W(data_align,S,W,train_mb,test_mb,d,model)
			transformed_pred.append(ut.transform(data_pred[d],W_tmp,model))
			# W_all.append(W_tmp)
			loo.append(loo_tmp)
		# # save W
		# if not os.path.exists(options['output_path']+'W/'+pre+'mapping{}/'.format(word_dim)+model+'/'):
		# 	os.makedirs(options['output_path']+'W/'+pre+'mapping{}/'.format(word_dim)+model+'/')
		# with open(options['output_path']+'W/'+pre+'mapping{}/'+model+'/'+'{}_feat{}_ntrain{}_rand{}_{}_ds{}.pickle'.format(word_dim,roi,nfeature,num_train,initseed,expopt,ds),'wb') as fid:
		# 	pickle.dump(W_all,fid,pickle.HIGHEST_PROTOCOL)
		# del W_all
	else:
		# average alignment data of training subjects as transformed alignment data
		S = []
		loo = []
		for d in range(ndata):
			train_subj,test_subj = ut.find_train_test_subj(train_mb,test_mb,d)
			S.append(np.mean(data_align[d][:,:,train_subj],axis=2))
			loo.append(test_subj)
		transformed_pred = data_pred

	print ('learn linear mapping')
	# learn linear mapping using:
	# 1) concatenated transformed alignment data from training subjects
	# 2) word embeddings for alignment data
	# process word embeddings
	word_align_all,word_pred_all = pred.process_semantic_all([word_align[n] for n in new_ds],[word_pred[n] for n in new_ds],num_previous)
	W_ft = pred.learn_linear_map([S[n] for n in new_ds],word_align_all,num_previous)

	print ('run experiment')
	if loo_flag:
		accu_class,accu_rank = pred.predict_loo([transformed_pred[n] for n in new_ds],word_pred_all,W_ft,[loo[n] for n in new_ds],num_chunks,num_previous)
	else:
		accu_class,accu_rank = pred.predict([transformed_pred[n] for n in new_ds],word_pred_all,W_ft,num_chunks,num_previous)

	print ('results:')
	print ('accu_class: '+str(accu_class))
	print ('accu_rank: '+str(accu_rank))
	# save results
	if not os.path.exists(options['output_path']+'accu/mapping/'+model+'/'):
		os.makedirs(options['output_path']+'accu/mapping/'+model+'/')
	np.savez_compressed(options['output_path']+'accu/mapping/'+model+'/{}_feat{}_rand{}_{}_ds{}.npz'.format(roi,nfeature,initseed,expopt,ds),accu_class=accu_class,accu_rank=accu_rank)


