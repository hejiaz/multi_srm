#!/usr/bin/env python
# Run correlation comparison experiment using SRM and dictlearn
# randomly partition subjects into two groups. Use half movie to train separately, and compute correlation 
# of the transformed the other half. Then train all subjects together, and compute correlation of the transformed
# the other half
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

# ds is the testing dataset
# save two correlation numbers:
# sep: correlation if trained separately
# tgr: correlation if trained together
def run_expt(nfeature,initseed,expopt,model,roi,ds):
	# parameters
	expt = 'corr'
	niter = 25

	print (model)
	print (roi)

	# import alignment and experiment method
	if model in ['indv_srm']:
		align = importlib.import_module('model.srm')
	elif model in ['indv_ica']:
		align = importlib.import_module('model.ica')
	elif model in ['indv_gica']:
		align = importlib.import_module('model.gica')
	elif model in ['indv_dict']:
		align = importlib.import_module('model.dictlearn')		
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

	# load input data
	print ('load data')
	with open(options['input_path']+'multi_srm/{}_data_all.pickle'.format(roi),'rb') as fid:
	    data_tmp = pickle.load(fid)

	data = data_tmp[ds] # data is a 3d array (voxel x time x nsubjs)
	del data_tmp

	voxel,nTR,nsubjs = data.shape
	# make sure there are equal number of TRs and subjects in each half
	if nTR%2:
		data = data[:,:-1,:]
		nTR = nTR-1
	if nsubjs%2:
		data = data[:,:,:-1]
		nsubjs = nsubjs-1

	# separate alignment and prediction data
	data_1st = ut.zscore_data_all(data[:,:int(nTR/2),:])
	data_2nd = ut.zscore_data_all(data[:,int(nTR/2):,:])
	if expopt == '1st':
		data_align = data_2nd
		data_pred = data_1st
	elif expopt == '2nd':
		data_align = data_1st
		data_pred = data_2nd
	else:
		raise Exception('expopt has to be 1st or 2nd')
	del data

	train_all = np.array(list(range(nsubjs)),dtype=np.int32).reshape(nsubjs,1)
	# randomly partition subjects into two groups
	data_align_sep = []
	train_mb = []
	new_order = list(range(nsubjs))
	random.seed(initseed)
	random.shuffle(new_order)
	# group 1
	grp1 = new_order[:int(nsubjs/2)]
	data_align_sep.append(data_align[:,:,grp1]) 
	train_mb.append(np.array(list(range(int(nsubjs/2))),dtype=np.int32).reshape(int(nsubjs/2),1))
	# group 2
	grp2 = new_order[int(nsubjs/2):]
	data_align_sep.append(data_align[:,:,grp2]) 
	train_mb.append(np.array(list(range(int(nsubjs/2))),dtype=np.int32).reshape(int(nsubjs/2),1))

	# load location information for dictionary learning
	if model in ['indv_dict']:
		ws = np.load(options['input_path']+'multi_srm/roi_location.npz')
		loc = ws[roi]
		del ws

	print ('alignment')
	if model not in ['avg']:
		# alignment
		# jointly learn using all subjects
		if model in ['indv_dict']:
			_,W_all,_= align.align([data_align],train_all,niter,nfeature,initseed,model,loc)
		else:
			W_all,_ = align.align([data_align],train_all,niter,nfeature,initseed,model)
		W_all = W_all[0]
		transformed_all = []
		# transform data for 2 groups 
		# group 1
		transformed_all.append(ut.transform(data_pred[:,:,grp1],W_all[:,:,grp1],model))
		# group 2
		transformed_all.append(ut.transform(data_pred[:,:,grp2],W_all[:,:,grp2],model))
		tgr = pred.predict(transformed_all)

		# learn using two groups separately
		if model in ['indv_dict']:
			_,W_grp1,_= align.align([data_align_sep[0]],train_mb[0],niter,nfeature,initseed,model,loc)
			_,W_grp2,_= align.align([data_align_sep[1]],train_mb[1],niter,nfeature,initseed,model,loc)
		else:
			W_grp1,_ = align.align([data_align_sep[0]],train_mb[0],niter,nfeature,initseed,model)
			W_grp2,_ = align.align([data_align_sep[1]],train_mb[1],niter,nfeature,initseed,model)
		W_grp1 = W_grp1[0]
		W_grp2 = W_grp2[0]
		transformed_grp = []
		# transform data for 2 groups
		# group 1
		transformed_grp.append(ut.transform(data_pred[:,:,grp1],W_grp1,model))
		# group 2
		transformed_grp.append(ut.transform(data_pred[:,:,grp2],W_grp2,model))
		sep = pred.predict(transformed_grp)
	else:
		transformed_all = []
		transformed_all.append(data_pred[:,:,grp1])
		transformed_all.append(data_pred[:,:,grp2])
		tgr = pred.predict(transformed_all)
		sep = tgr


	print ('sep: '+str(sep))
	print ('tgr: '+str(tgr))
	# save results
	if not os.path.exists(options['output_path']+'accu/corr/'+model+'/'):
		os.makedirs(options['output_path']+'accu/corr/'+model+'/')
	out_file = options['output_path']+'accu/corr/'+model+'/'+'{}_feat{}_rand{}_{}_ds{}.npz'.format(roi,nfeature,initseed,expopt,ds)
	np.savez_compressed(out_file,sep=sep,tgr=tgr)

