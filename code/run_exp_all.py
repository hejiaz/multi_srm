#!/usr/bin/env python
# run experiments with different parameters
# By Hejia Zhang @ Princeton


import argparse
import importlib
import sys, os
sys.path.insert(0, os.path.abspath('..'))

## argument parsing
usage = '%(prog)s expt model nfeat'
parser = argparse.ArgumentParser(usage=usage)

parser.add_argument("expt",    help="which experiment to run")
parser.add_argument("model",   help="which model to use, can be 'all_xxx' or 'indv_xxx' or 'multi_srm' ")  
parser.add_argument("nfeat",   type=int, help="number of features")
args = parser.parse_args()

# experiments parameters
num_train = 40
loo_flag = True
ds = [0,1,2,3] # which datasets to use: greeneye,milky,vodka,sherlock
expopt_all = ['1st','2nd']
word_dim = 300
loo_ds_all = [1,2,3]

if args.expt in ['loods']:
	loo_all = [0,3,1]
	other_all = [[1,3],[1,0],[0,3]]

# import experiment 
run = importlib.import_module('run_'+args.expt)

if args.expt in ['mysseg','mysseg_all']:
	# for initseed in [0, 1, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 24, 25, 26, 27]:
	for initseed in [0, 1, 3]:
		for expopt in expopt_all:
			for roi in ['dmn','pt','eac']:
				run.run_expt(args.nfeat,initseed,expopt,num_train,loo_flag,args.model,roi,ds)

elif args.expt in ['mapping']:
	# for initseed in [0, 1, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 24, 25, 26, 27]:
	for initseed in [0, 1, 3, 5, 6]:
		for expopt in expopt_all:
			for roi in ['dmn','pt','eac']:
				run.run_expt(args.nfeat,initseed,expopt,word_dim,num_train,loo_flag,args.model,roi,ds)

elif args.expt in ['mapping_loo']:
	for initseed in range(5):
		for loo_ds in loo_ds_all:
			for roi in ['dmn','pt','eac']:
				run.run_expt(args.nfeat,initseed,word_dim,args.model,roi,loo_ds)

elif args.expt in ['loods']:
	for initseed in range(5):
		for loo_ds,other_ds in zip(loo_all,other_all):
			for roi in ['dmn','pt','eac']:
				run.run_expt(args.nfeat,initseed,roi,loo_ds,other_ds)



