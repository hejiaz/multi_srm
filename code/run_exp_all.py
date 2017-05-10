#!/usr/bin/env python
# run experiments with different parameters
# By Hejia Zhang @ Princeton


import argparse
import importlib
import sys, os
sys.path.insert(0, os.path.abspath('..'))

## argument parsing
usage = '%(prog)s expt model rand'
parser = argparse.ArgumentParser(usage=usage)

parser.add_argument("expt",    help="which experiment to run")
parser.add_argument("model",   help="which model to use, can be 'all_xxx' or 'indv_xxx' or 'multi_srm' ")  
parser.add_argument("rand",   type=int, help="random seed to use")
parser.add_argument("-s","--shared",   type=int, help="number of shared subjects")
parser.add_argument("-d","--ds", type = lambda s: [int(item) for item in s.split(',')], help="ds in a list format (e.g.: '1,2,3')")
args = parser.parse_args()

# initseed = [0, 1, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 24, 25, 26, 27]
# initseed = [0, 2, 6, 7, 8, 9, 15, 25, 26, 27, 28, 29] # for all ds

# experiments parameters
num_train = 65 #20 test
ds = list(range(12))

# num_train = 40
# ds = [0,1,2,3] # which datasets to use: greeneye,milky,vodka,sherlock

loo_flag = True
expopt_all = ['1st','2nd']

if args.expt in ['mapping_loo']:
	loo_ds_all = [1,2,3]

if args.expt in ['loods']:
	loo_all = [0,3,1]
	other_all = [[1,3],[1,0],[0,3]]

if args.expt in ['overfit']:
	ds_all = [[0,1],[0,2],[0,3],[1,3]]

if args.expt in ['mapping_all']:
	tst_ds_all = [3]

if args.expt in ['shared_subj','dist_subj']:
	# ds_all = [[0,1],[1,0],[0,3],[3,0],[1,3],[3,1],[0,2],[2,0]]
	ds_all = [[3,0]]

# import experiment 
run = importlib.import_module('run_'+args.expt)

if args.expt in ['mysseg','mysseg_all']:
	feat_dict = {'multi_srm':[75,75,100],'multi_dict':[25,50,50],'indv_srm':[75,75,50],'all_srm':[75,75,50],'indv_ica':[50,25,25],\
	'indv_gica':[50,50,25],'indv_dict':[50,25,50],'avg':[50,50,50]}
	for expopt in expopt_all:
		for roi,nfeat in zip(['dmn','pt','eac'],feat_dict[args.model]):
			try:
				run.run_expt(nfeat,args.rand,expopt,num_train,loo_flag,args.model,roi,ds)
			except:
				print (sys.exc_info()[0])
				continue

elif args.expt in ['mapping']:
	feat_dict = {'multi_srm':[75,150,75],'multi_dict':[75,50,50],'indv_srm':[75,50,75],'all_srm':[75,50,75],'indv_ica':[50,50,25],\
	 'indv_gica':[50,50,50],'indv_dict':[50,25,50],'avg':[50,50,50]}
	for expopt in expopt_all:
		for roi,nfeat in zip(['dmn','pt','eac'],feat_dict[args.model]):
			try:
				run.run_expt(nfeat,args.rand,expopt,num_train,loo_flag,args.model,roi,ds)
			except:
				print (sys.exc_info()[0])
				continue

elif args.expt in ['imgpred']:
	feat_dict = {'multi_srm':[125],'multi_dict':[100],'indv_srm':[125],'all_srm':[125],'indv_ica':[25],\
	'indv_dict':[100], 'indv_gica':[25],'avg':[50]}
	for roi,nfeat in zip(['pmc'],feat_dict[args.model]):
		try:
			run.run_expt(nfeat,args.rand,loo_flag,args.model,roi,ds)
		except:
			print (sys.exc_info()[0])
			continue

elif args.expt in ['mapping_loo']:
	feat_dict = {'multi_srm':[75,150,75],'multi_dict':[75,50,50],'indv_srm':[75,50,75],'indv_ica':[75,50,25],\
	 'indv_gica':[75,50,50],'indv_dict':[50,25,50],'avg':[50,50,50]}
	for loo_ds in loo_ds_all:
		for roi,nfeat in zip(['dmn','pt','eac'],feat_dict[args.model]):
			try:
				run.run_expt(nfeat,args.rand,args.model,roi,loo_ds)
			except:
				print (sys.exc_info()[0])
				continue

elif args.expt in ['mapping_all']:
	feat_dict = {'multi_srm':[75,150,75],'multi_dict':[75,50,50],'indv_srm':[75,50,75],'indv_ica':[75,50,25],\
	 'indv_gica':[75,50,50],'indv_dict':[50,25,50],'avg':[50,50,50]}
	for tst_ds in tst_ds_all:
		for expopt in expopt_all:
			for roi,nfeat in zip(['dmn','pt','eac'],feat_dict[args.model]):
				try:
					run.run_expt(nfeat,args.rand,expopt,num_train,loo_flag,args.model,roi,ds,tst_ds)
				except:
					print (sys.exc_info()[0])
					continue

elif args.expt in ['loods']:
	feat_dict = {'multi_srm':[75,75,100],'multi_dict':[25,50,50],'indv_srm':[75,75,50],'indv_ica':[50,25,25],\
	'all_gica':[100,50,25],'indv_dict':[50,25,50],'avg':[50,50,50]}
	for loo_ds,other_ds in zip(loo_all,other_all):
		for roi,nfeat in zip(['dmn','pt','eac'],feat_dict[args.model]):
			try:
				run.run_expt(nfeat,args.rand,args.model,roi,loo_ds,other_ds)
			except:
				print (sys.exc_info()[0])
				continue

elif args.expt in ['overfit']:
	feat_dict = {'multi_srm':[75,75,100],'multi_dict':[25,50,50],'indv_srm':[75,75,50],'indv_ica':[50,25,25],\
	'all_gica':[100,50,25],'indv_dict':[50,25,50],'avg':[50,50,50]}
	for expopt in expopt_all:
		for ds in ds_all:
			for roi,nfeat in zip(['dmn','pt','eac'],feat_dict[args.model]):
				try:
					run.run_expt(nfeat,args.rand,expopt,args.model,roi,ds)
				except:
					print (sys.exc_info()[0])
					continue

elif args.expt in ['shared_subj']:
	feat_dict = {'multi_srm':[75,75,100],'multi_dict':[25,50,50]}
	for ds in ds_all:
		# for roi,nfeat in zip(['dmn','pt','eac'],feat_dict[args.model]):
		for roi,nfeat in zip(['dmn'],[feat_dict[args.model][0]]):
			for expopt in expopt_all:
				try:
					run.run_expt(nfeat,args.rand,expopt,args.model,roi,ds)
				except:
					print (sys.exc_info()[0])
					continue

elif args.expt in ['dist_subj']:
	feat_dict = {'multi_srm':[75,75,100],'multi_dict':[25,50,50]}
	# for ds in ds_all:
	# for roi,nfeat in zip(['dmn','pt','eac'],feat_dict[args.model]):
	for roi,nfeat in zip(['dmn'],[feat_dict[args.model][0]]):
		for expopt in expopt_all:
			try:
				run.run_expt(nfeat,args.rand,expopt,args.model,roi,args.ds,args.shared)
			except:
				print (sys.exc_info()[0])
				continue

