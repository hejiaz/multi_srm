#!/usr/bin/env python
# run experiments with different parameters
# By Hejia Zhang @ Princeton

from run_mysseg import run_mysseg_expt

# mysseg experiments
# num_train = 40
# loo_flag = True
# ds = [0,1,2,3] # which datasets to use: greeneye,milky,vodka,sherlock
# for nfeature in [25,50]:
# 	# for initseed in [0, 1, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20, 21, 24, 25, 26, 27]:
# 	for initseed in [0, 1, 3, 5, 6]:
# 		for expopt in ['1st','2nd']:
# 			# for model in ['multi_srm','srm_rotate','srm_rotate_ind','indv_srm','avg']:
# 			for model in ['indv_srm']:
# 				for roi in ['dmn','pt','eac']:
# 					run_mysseg_expt(nfeature,initseed,expopt,num_train,loo_flag,model,roi,ds)


num_train = 10
loo_flag = True
ds = [1]
nfeature = 10
initseed = 0
expopt = '1st'
# model = 'srm'
model = 'indv_srm'
roi = 'pt'

run_mysseg_expt(nfeature,initseed,expopt,num_train,loo_flag,model,roi,ds)