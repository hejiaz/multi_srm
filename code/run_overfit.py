#!/usr/bin/env python
# For two datasets A and B. Use first half of each dataset to learn a set of W (W_A, W_B), and test on second half of
# shared subjects (X_A, X_B) using W_A and W_B. X_A using W_A,W_B; X_B using W_A, W_B.
# By Hejia Zhang @ Princeton

import numpy as np
import yaml
import os
import pickle
import importlib
import utils as ut
import sys, os
sys.path.insert(0, os.path.abspath('..'))

# ds: which two datasets to compare
# accu1,accu2: 2d arrays (# shared subjects x 2). [:,0]: test using its own W; [:,1]: test using W learned from the other dataset
def run_expt(nfeature,initseed,expopt,model,roi,ds):
	# parameters
	expt = 'mysseg'
	niter = 50

	print (roi)
	print ('ds1: '+str(ds[0])+', ds2: '+str(ds[1]))

	# import alignment and experiment method
	if model in ['indv_srm']:
		align = importlib.import_module('model.srm')
	elif model in ['indv_ica']:
		align = importlib.import_module('model.ica')
	elif model in ['indv_gica']:
		align = importlib.import_module('model.gica')
	elif model in ['indv_dict']:
		align = importlib.import_module('model.dictlearn')
	else:
		raise Exception('invalid model')
	
	pred = importlib.import_module('experiment.'+expt)

	# load path
	# setting = open('setting.yaml')
	setting = open('../setting.yaml')
	options = yaml.safe_load(setting)

	# load location information for dictionary learning
	if model in ['indv_dict']:
		ws = np.load(options['input_path']+'multi_srm/roi_location.npz')
		loc = ws[roi]
		del ws

	# load membership info
	ws = np.load(options['input_path']+'multi_srm/membership.npz')
	membership = ws['membership']

	# extract datasets we want to use
	membership = membership[:,ds]
	membership = ut.remove_invalid_membership(membership)

	# check if there are shared subjects between datasets
	if not ut.check_membership(membership):
		raise Exception('No shared subjects')

	# load input data
	print ('load data')
	with open(options['input_path']+'multi_srm/{}_data_all.pickle'.format(roi),'rb') as fid:
	    data_tmp = pickle.load(fid)

	# extract data in dataset1
	TR1st = int(data_tmp[ds[0]].shape[1]/2)
	data1_1st = [ut.zscore_data_all(data_tmp[ds[0]][:,:TR1st,:])]
	data1_2nd = [ut.zscore_data_all(data_tmp[ds[0]][:,TR1st:,:])]
	# extract data in dataset2
	TR1st = int(data_tmp[ds[1]].shape[1]/2)
	data2_1st = [ut.zscore_data_all(data_tmp[ds[1]][:,:TR1st,:])]
	data2_2nd = [ut.zscore_data_all(data_tmp[ds[1]][:,TR1st:,:])]
	# separate align and pred
	if expopt == '1st':
		data1_align = data1_2nd
		data1_pred = data1_1st
		data2_align = data2_2nd
		data2_pred = data2_1st
	elif expopt == '2nd':
		data1_align = data1_1st
		data1_pred = data1_2nd
		data2_align = data2_1st
		data2_pred = data2_2nd
	else:
		raise Exception('expopt has to be 1st or 2nd')
	del data_tmp

	# find indices of shared subjects between dataset1 and dataset2
	idx1,idx2 = ut.find_shared_subjects_between_two(membership,0,1)

	# extract membership info for dataset1 and 2 separately
	mb1 = membership[:,0][:,None]
	mb1 = ut.remove_invalid_membership(mb1)
	mb2 = membership[:,1][:,None]
	mb2 = ut.remove_invalid_membership(mb2)
	test_mb = np.empty((0),dtype=np.int32)

	# alignment for each dataset
	if model in ['indv_dict']:
		_,W1,S= align.align(data1_align,mb1,niter,nfeature,initseed,model,loc)
		_,W2,_= align.align(data2_align,mb2,niter,nfeature,initseed,model,loc)
	else:
		W1,S = align.align(data1_align,mb1,niter,nfeature,initseed,model)	
		W2,_ = align.align(data2_align,mb2,niter,nfeature,initseed,model)

	# reorder W so that it aligns with order in data
	W1,_= ut.learn_W(data1_align,S,W1,mb1,test_mb,0,model)
	W2,_= ut.learn_W(data2_align,S,W2,mb2,test_mb,0,model)

	# extract data and W that corresponds to shared subjects
	X1 = data1_pred[0][:,:,idx1]
	X2 = data2_pred[0][:,:,idx2]
	W1 = W1[:,:,idx1]
	W2 = W2[:,:,idx2]

	# initialize accuracy array
	accu1 = np.zeros((len(idx1),2),dtype=np.float32)
	accu2 = np.zeros((len(idx1),2),dtype=np.float32)

	# experiment on 4 combinations
	transformed_pred = ut.transform(X1,W1,model)
	accu1[:,0] = pred.predict(transformed_pred)
	print ('accu1,own: '+str(np.mean(accu1[:,0])))

	transformed_pred = ut.transform(X1,W2,model)
	accu1[:,1] = pred.predict(transformed_pred)
	print ('accu1,other: '+str(np.mean(accu1[:,1])))

	transformed_pred = ut.transform(X2,W2,model)
	accu2[:,0] = pred.predict(transformed_pred)
	print ('accu2,own: '+str(np.mean(accu2[:,0])))	

	transformed_pred = ut.transform(X2,W1,model)
	accu2[:,1] = pred.predict(transformed_pred)
	print ('accu2,other: '+str(np.mean(accu2[:,1])))

	# save results
	if not os.path.exists(options['output_path']+'accu/overfit/'+model+'/'):
		os.makedirs(options['output_path']+'accu/overfit/'+model+'/')
	out_file = options['output_path']+'accu/overfit/'+model+'/'+'{}_feat{}_rand{}_{}_ds{}.npz'.format(roi,nfeature,initseed,expopt,ds).replace(' ','')
	np.savez_compressed(out_file,accu1=accu1,accu2=accu2)

