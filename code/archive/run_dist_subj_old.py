#!/usr/bin/env python
# For two datasets A and B ([0,1] or [0,2]), use all subjects in A and shared subjects in B and s distinct subjects in B
# to train, use all subjects in A to test. One half (in TR) to train, the other half to test. Try all possible number 
# of distinct subjects.
# By Hejia Zhang @ Princeton

import numpy as np
import yaml,copy
import pickle
import random
import importlib
import utils as ut
import sys, os
sys.path.insert(0, os.path.abspath('..'))

# ds: which two datasets to use
# accu: a list of length (# distinct subjects in B, compared with A)
def run_expt(nfeature,initseed,expopt,model,roi,ds):
	# parameters
	expt = 'mysseg'
	niter = 50

	print (roi)
	print ('ds1: '+str(ds[0])+', ds2: '+str(ds[1]))

	# import alignment and experiment method
	if model in ['multi_srm']:
		align = importlib.import_module('model.multi_srm')
	elif model in ['multi_dict']:
		align = importlib.import_module('model.dictlearn')
	else:
		raise Exception('invalid model')
	
	pred = importlib.import_module('experiment.'+expt)

	# load path
	try:
		setting = open('setting.yaml')
	except:
		setting = open('../setting.yaml')
	options = yaml.safe_load(setting)

	# load location information for dictionary learning
	if model in ['multi_dict']:
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

	# extract data in 2 datasets
	TR_1st = int(data_tmp[ds[0]].shape[1]/2)

	data_align = []
	data_pred = []
	# use half TR of dataset 1 for alignment, the other half for prediction
	if expopt == '1st':
		data_align.append(ut.zscore_data_all(data_tmp[ds[0]][:,:TR_1st,:]))
		data_pred.append(ut.zscore_data_all(data_tmp[ds[0]][:,TR_1st:,:]))
	elif expopt == '2nd':
		data_pred.append(ut.zscore_data_all(data_tmp[ds[0]][:,:TR_1st,:]))
		data_align.append(ut.zscore_data_all(data_tmp[ds[0]][:,TR_1st:,:]))	
	else:
		raise Exception('expopt has to be 1st or 2nd')
	# use all TR of dataset 2 for alignment
	data_align.append(ut.zscore_data_all(data_tmp[ds[1]]))
	data_pred.append(None)
	del data_tmp

	# find indices of shared subjects between dataset1 and dataset2
	idx1,idx2 = ut.find_shared_subjects_between_two(membership,0,1) #for ds[0] and ds[1]
	num_ds0 = np.count_nonzero(membership[:,0] != -1) # total number of subjects in ds[0]
	num_ds1 = np.count_nonzero(membership[:,1] != -1) # total number of subjects in ds[1]
	idx_diff = [n for n in list(range(num_ds1)) if n not in idx2] # index of distinct subjects in ds[1]

	# find a random order to add distinct subjects
	new_order = list(range(len(idx_diff)))
	random.seed(initseed)
	random.shuffle(new_order)

	# initialization
	accu = []
	new_mb = copy.copy(membership)

	# test on all data first
	# find global indices of all subjects in ds[0]	
	idx_all = []
	for i in range(num_ds0):
		idx_all.append(list(new_mb[:,0]).index(i))

	# alignment
	if model in ['multi_srm']:
		W,S,_ = align.align(data_align,new_mb,niter,nfeature,initseed,model)
	elif model in ['multi_dict']:
		W_grp,W,S= align.align(data_align,new_mb,niter,nfeature,initseed,model,loc)
	else:
		raise Exception('invalid model')
	# extract W in the same order as ds[0]
	W1 = W[:,:,idx_all]
	# transform
	transformed_pred = ut.transform(data_pred[0],W1,model)
	# experiment
	accu_m = pred.predict(transformed_pred)
	accu.append(np.mean(accu_m))
	print (np.mean(accu_m))

	# loop over different number of distinct subjects by removing them one by one
	for m in range(len(idx_diff)):
		# find the next removed distinct subject
		new_subj = list(new_mb[:,1]).index(idx_diff[new_order[m]])
		new_mb[new_subj,1] = -1
		new_mb = ut.remove_invalid_membership(new_mb)

		# find global indices of all subjects in ds[0]	
		idx_all = []
		for i in range(num_ds0):
			idx_all.append(list(new_mb[:,0]).index(i))

		# alignment
		if model in ['multi_srm']:
			W,S,_ = align.align(data_align,new_mb,niter,nfeature,initseed,model)
		elif model in ['multi_dict']:
			W_grp,W,S= align.align(data_align,new_mb,niter,nfeature,initseed,model,loc)
		else:
			raise Exception('invalid model')
		# extract W in the same order as ds[0]
		W1 = W[:,:,idx_all]
		# transform
		transformed_pred = ut.transform(data_pred[0],W1,model)
		# experiment
		accu_m = pred.predict(transformed_pred)
		accu.append(np.mean(accu_m))
		print (np.mean(accu_m))


	# reverse order
	accu.reverse()

	# save results
	if not os.path.exists(options['output_path']+'accu/dist_subj/'+model+'/'):
		os.makedirs(options['output_path']+'accu/dist_subj/'+model+'/')
	out_file = options['output_path']+'accu/dist_subj/'+model+'/'+'{}_feat{}_rand{}_{}_ds{}.npz'.format(roi,nfeature,initseed,expopt,ds)
	np.savez_compressed(out_file,accu=np.array(accu,dtype=np.float32))

