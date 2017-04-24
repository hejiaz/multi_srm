#!/usr/bin/env python
# run experiments with different parameters
# By Hejia Zhang @ Princeton

from left_out_dataset import left_out_dataset_expt

# dataset pairs without shared subjects: [1, 2], [2, 3]

# left_out_dataset experiments
nfeature = 50
loo_ds_all = [0,0,1,1,1,2,2,3,3,3] 
other_ds_all = [[1,3],[3,1],[0,2,3],[0,3,2],[3,0,2],[0,1,3],[0,3,1],[0,1,2],[0,2,1],[1,0,2]]
for initseed in range(5):
	for loo_ds,other_ds in zip(loo_ds_all,other_ds_all):
		for roi in ['dmn','pt','eac']:
			left_out_dataset_expt(nfeature,initseed,roi,loo_ds,other_ds)


