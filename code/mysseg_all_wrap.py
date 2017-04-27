#!/usr/bin/env python
# run experiments with different parameters
# By Hejia Zhang @ Princeton

from run_mysseg_all import run_mysseg_all_expt

# mysseg_all experiments
num_train = 40
loo_flag = True
ds = [0,1,2,3] # which datasets to use: greeneye,milky,vodka,sherlock
for nfeature in [25,50]:
	# for initseed in [0, 1, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 24, 25, 26, 27]:
	for initseed in [0, 1, 3, 5, 6]:
		for expopt in ['1st','2nd']:
			# for model in ['multi_srm','srm_rotate','srm_rotate_ind','indv_srm','avg']:
			for model in ['indv_srm']:
				for roi in ['dmn','pt','eac']:
					run_mysseg_all_expt(nfeature,initseed,expopt,num_train,loo_flag,model,roi,ds)

