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
parser.add_argument("nfeat",   type=int, help="random seed to use")
parser.add_argument("roi",   help="which roi to use") 
args = parser.parse_args()

# initseed = [0, 1, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 24, 25, 26, 27]

# experiments parameters
# num_train = 40
# ds = [0,1,2,3] # which datasets to use: greeneye,milky,vodka,sherlock

num_train = 65
ds = list(range(12))

loo_flag = True
# word_ds = [0,1,2,3] # which datasets has text embeddings, only use these datasets to test mapping; can use others to help learn W and S
expopt_all = ['1st','2nd']

# import experiment 
run = importlib.import_module('run_'+args.expt)

if args.expt in ['mysseg']:
	for rand in [0,2,6]:
		for expopt in expopt_all:
			run.run_expt(args.nfeat,rand,expopt,num_train,loo_flag,args.model,args.roi,ds)

elif args.expt in ['mapping']:
	for rand in [0,2,6]:
		for expopt in expopt_all:
			run.run_expt(args.nfeat,rand,expopt,num_train,loo_flag,args.model,args.roi,ds)

elif args.expt in ['imgpred']:
	for rand in [0,1,2]:
		run.run_expt(args.nfeat,rand,loo_flag,args.model,args.roi,ds)			


