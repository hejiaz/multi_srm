#!/usr/bin/env python
# Choose which random seed gives meaningful separation of training and testing subejcts in loo experiments
# Depends on which datasets to include, and how many number of training subjects. 
# For training: datasets should be able to be linked through shared training subjects
# For testing: each dataset has at least min_test testing subjects
# By Hejia Zhang @ Princeton

import numpy as np
import yaml
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import utils as ut
import random

# parameters
min_test = 3
num_train = 65
ds = list(range(12)) # which datasets to use: greeneye,milky,vodka,sherlock, plus 8 small datasets (12 in total)
max_rand = 30

# load membership info
setting = open('../setting.yaml')
options = yaml.safe_load(setting)
ws = np.load(options['input_path']+'multi_srm/membership.npz')
membership = ws['membership']

# extract datasets we want
membership = membership[:,ds]
membership = ut.remove_invalid_membership(membership)
nsubjs = membership.shape[0]

# test all random seeds less than max_rand
rand_working = []
for initseed in range(max_rand):
	# separate training and testing subjects
	new_order = list(range(nsubjs))
	random.seed(initseed)
	random.shuffle(new_order)
	new_mb = np.array([membership[n,:] for n in new_order])
	train_mb = new_mb[:num_train,:]
	test_mb = new_mb[num_train:,:]
	# check if the training subjects can be linked through shared subjects
	if not ut.check_membership(train_mb):
		continue
	# check if all datasets have at least min_test testing subjects
	num_subj = ut.count_num_subject_in_each_dataset(test_mb)
	if all(n>=min_test for n in num_subj):
		rand_working.append(initseed)

print (rand_working)

