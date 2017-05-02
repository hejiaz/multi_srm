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
args = parser.parse_args()

# initseed = [0, 1, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 24, 25, 26, 27]

# experiments parameters
num_train = 40
loo_flag = True
# word_ds = [0,1,2,3] # which datasets has text embeddings, only use these datasets to test mapping; can use others to help learn W and S
ds = [0,1,2,3] # which datasets to use: greeneye,milky,vodka,sherlock
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

# import experiment 
run = importlib.import_module('run_'+args.expt)

if args.expt in ['mysseg','mysseg_all']:
	feat_dict = {'multi_srm':[75,75,100],'all_srm':[75,75,50],'indv_srm':[75,75,50],'all_ica':[50,25,25],'indv_ica':[50,25,25],\
	'all_gica':[100,50,25],'indv_gica':[100,50,25],'all_dict':[25,25,25],'indv_dict':[25,25,25],'avg':[50,50,50]}
	for expopt in expopt_all:
		for roi,nfeat in zip(['dmn','pt','eac'],feat_dict[args.model]):
			run.run_expt(nfeat,args.rand,expopt,num_train,loo_flag,args.model,roi,ds)

elif args.expt in ['mapping']:
	feat_dict = {'multi_srm':[200,50,150],'all_srm':[50,125,50],'all_ica':[25,25,25],'all_gica':[75,50,75],'all_dict':[200,150,75],'avg':[50,50,50]}
	for expopt in expopt_all:
		for roi,nfeat in zip(['dmn','pt','eac'],feat_dict[args.model]):
			run.run_expt(nfeat,args.rand,expopt,num_train,loo_flag,args.model,roi,ds)

elif args.expt in ['mapping_loo']:
	feat_dict = {'multi_srm':[200,50,150],'all_srm':[50,125,50],'all_ica':[25,25,25],'all_gica':[75,50,75],'all_dict':[200,150,75],'avg':[50,50,50]}
	for loo_ds in loo_ds_all:
		for roi,nfeat in zip(['dmn','pt','eac'],feat_dict[args.model]):
			run.run_expt(nfeat,args.rand,args.model,roi,loo_ds)

elif args.expt in ['mapping_all']:
	feat_dict = {'multi_srm':[200,50,150],'all_srm':[50,125,50],'all_ica':[25,25,25],'all_gica':[75,50,75],'all_dict':[200,150,75],'avg':[50,50,50]}
	for tst_ds in tst_ds_all:
		for expopt in expopt_all:
			for roi,nfeat in zip(['dmn','pt','eac'],feat_dict[args.model]):
				run.run_expt(nfeat,args.rand,expopt,num_train,loo_flag,args.model,roi,ds,tst_ds)

elif args.expt in ['loods']:
	feat_dict = {'multi_srm':[75,75,100],'all_srm':[75,75,50],'indv_srm':[75,75,50],'all_ica':[50,25,25],'indv_ica':[50,25,25],\
	'all_gica':[100,50,25],'indv_gica':[100,50,25],'all_dict':[25,25,25],'indv_dict':[25,25,25],'avg':[50,50,50]}
	for loo_ds,other_ds in zip(loo_all,other_all):
		for roi,nfeat in zip(['dmn','pt','eac'],feat_dict[args.model]):
			run.run_expt(nfeat,args.rand,roi,loo_ds,other_ds)

elif args.expt in ['overfit']:
	feat_dict = {'indv_srm':[75,75,50],'indv_ica':[50,25,25],'indv_gica':[100,50,25],'indv_dict':[25,25,25]}
	for expopt in expopt_all:
		for ds in ds_all:
			for roi,nfeat in zip(['dmn','pt','eac'],feat_dict[args.model]):
				run.run_expt(nfeat,args.rand,expopt,args.model,roi,ds)




