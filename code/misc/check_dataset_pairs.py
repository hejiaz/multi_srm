#!/usr/bin/env python
# Check all dataset pairs and output which ones do not have shared subjects
# By Hejia Zhang @ Princeton

import numpy as np
import yaml
import utils as ut
import itertools

# dataset code: greeneye,milky,vodka,sherlock
ds_all = [0,1,2,3]

# load membership info
setting = open('setting.yaml')
options = yaml.safe_load(setting)
ws = np.load(options['input_path']+'multi_srm/membership.npz')
membership = ws['membership']

combo = itertools.combinations(ds_all,2)
not_working = []
for ds in combo:
	# extract datasets we want
	new_mb = membership[:,list(ds)]
	new_mb = ut.remove_invalid_membership(new_mb)
	# check if the training subjects can be linked through shared subjects
	if not ut.check_membership(new_mb):
		not_working.append(list(ds))
print (not_working)

# [[1, 2], [2, 3]]