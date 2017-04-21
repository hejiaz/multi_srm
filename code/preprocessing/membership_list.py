#!/usr/bin/env python

# By Hejia Zhang @ Princeton

import numpy as np
import csv
import yaml

# load path
setting = open('../setting.yaml')
options = yaml.safe_load(setting)

# load raw data and extract subject list (55 subjects in total)
data = []
with open(options['raw_path']+'multi_srm/shared_subjects.csv','r') as f:
	reader = csv.reader(f,delimiter=',')
	for row in reader:
		data.append(row)
ary = np.array(data)
ary[0,0] = ary[0,0][1:]

# subject_ary is a 55-by-4 (total number of subjects by total number of datasets) array
# if subject i is in dataset j: subject_ary[i,j] = subject id of subject i in dataset j
# if subject i is not in dataset j: subject_ary[i,j] = -1
subject_ary = np.zeros(ary.shape,dtype=np.int32)
for i in range(ary.shape[0]):
	for j in range(ary.shape[1]):
		if ary[i,j] == 'None':
			subject_ary[i,j] = -1
		else:
			subject_ary[i,j] = int(ary[i,j])

# save results
np.savez_compressed(options['input_path']+'multi_srm/membership.npz',membership=subject_ary)
print (options['input_path']+'multi_srm/membership.npz')