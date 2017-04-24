#!/usr/bin/env python
# run experiments with different parameters
# By Hejia Zhang @ Princeton

from run_mapping_loo import run_mapping_loo_expt

# mapping experiments
nfeature = 25
word_dim = 100
for initseed in range(5):
	for loo_ds in [1,2,3]:
		for model in ['multi_srm','srm_rotate','srm_rotate_ind','avg']:
			for roi in ['dmn','pt','eac']:
				run_mapping_loo_expt(nfeature,initseed,word_dim,model,roi,loo_ds)
