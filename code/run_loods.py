#!/usr/bin/env python
# Leave one dataset out, test the time segment matching accuracy using W learned from other datasets.
# Only test on shared subjects between loo_ds and other_ds[0](base dataset). Merge other datasets one-by-one 
# in the order of other_ds.
# By Hejia Zhang @ Princeton

import numpy as np
import yaml
import os
import pickle
import importlib
import utils as ut
import sys, os
sys.path.insert(0, os.path.abspath('..'))


# roi = 'dmn'
# # datasets code: greeneye,milky,vodka,sherlock
# loo_ds = 1 # left-out dataset, can't leave out 0
# other_ds = [0,2,3] # the merging order of other datasets

def run_expt(nfeature,initseed,model,roi,loo_ds,other_ds):
	# model: multi_srm or multi_dict
	# parameters
	expt = 'mysseg'
	niter = 50

	print (roi)
	print ('left-out: '+str(loo_ds)+', base: '+str(other_ds[0]))

	# import alignment and experiment method
	if model in ['multi_srm']:
		align = importlib.import_module('model.srm')
	elif model in ['multi_dict']:
		align = importlib.import_module('model.dictlearn')
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

	# load input data
	print ('load data')
	with open(options['input_path']+'multi_srm/{}_data_all.pickle'.format(roi),'rb') as fid:
	    data_tmp = pickle.load(fid)

	# extract data in left-out dataset
	data_test = data_tmp[loo_ds]

	# find shared subjects between left-out dataset and base dataset
	idx_loo,idx_base = ut.find_shared_subjects_between_two(membership,loo_ds,other_ds[0])

	# only keep shared subjects in left-out dataset
	data_test = data_test[:,:,idx_loo]

	# accu[d] is accuracy after adding first d+1 datasets in other_ds
	accu = []

	for d in range(len(other_ds)):
		print ('training datasets: '+str(other_ds[:d+1]))
		# extract membership info in this sub experiment
		sub_mb = membership[:,other_ds[:d+1]]
		sub_mb = ut.remove_invalid_membership(sub_mb)
		# check if the training subjects can be linked through shared subjects
		if not ut.check_membership(sub_mb):
			raise Exception('Not all datasets can be linked through shared training subjects')
		# extract datasets in this sub experiment
		data = []
		for i in range(d+1):
			data.append(data_tmp[other_ds[i]])
		# alignment
		W,_,_ = align.align(data,sub_mb,niter,nfeature,initseed,model)
		# extract W_loo: W of shared subjects
		idx_all = ut.map_ds_subject_idx_to_global(membership,other_ds[0],idx_base,other_ds[:d+1])
		W_loo = W[:,:,idx_all]
		# transform shared subjects in left-out dataset into shared space
		transformed_pred = ut.transform(data_test,W_loo,model)
		# experiment
		accu.append(pred.predict(transformed_pred))
		print ('accu'+str(d)+': '+str(np.mean(accu[d])))

	# save results
	if not os.path.exists(options['output_path']+'accu/loo_ds/'+model+'/'):
		os.makedirs(options['output_path']+'accu/loo_ds/'+model+'/')
	out_file = options['output_path']+'accu/loo_ds/'+model+'/'+'{}_feat{}_rand{}_loo{}_other{}.pickle'.format(roi,nfeature,initseed,loo_ds,other_ds)
	with open(out_file,'wb') as fid:
		pickle.dump(accu,fid,pickle.HIGHEST_PROTOCOL)

