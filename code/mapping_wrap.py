#!/usr/bin/env python
# run experiments with different parameters
# By Hejia Zhang @ Princeton

from run_mapping import run_mapping_expt

# mapping experiments
nfeature = 50
num_train = 40
loo_flag = True
ds = [0,1,2,3] # which datasets to use: greeneye,milky,vodka,sherlock
# for initseed in [0, 1, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 24, 25, 26, 27]:
for initseed in [0, 1, 3, 5, 6]:
	for expopt in ['1st','2nd']:
		for word_dim in [100]:
			for model in ['multi_srm','srm_rotate','srm_rotate_ind','indv_srm','avg']:
				for roi in ['dmn','pt','eac']:
					run_mapping_expt(nfeature,initseed,expopt,word_dim,num_train,loo_flag,model,roi,ds)
