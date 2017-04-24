#!/usr/bin/env python
import numpy as np
from numpy.linalg import norm
from sklearn.random_projection import GaussianRandomProjection
from scipy.io import loadmat
from sklearn.metrics.pairwise import pairwise_distances
import yaml
import os

# project word-embeddings from 300 dim to 100 dim
datasets = ['GreenEyes','milky','vodka','sherlock']
vec_name = ['uavgvecs','uavgvecs','uavgvecs','uavgvecs']
ndata = 4

setting = open('../setting.yaml')
options = yaml.safe_load(setting)
in_path = options['raw_path']+'wordembeddings/uavgvecs_{}.mat'
out_path = options['input_path']+'wordembeddings{}/{}.npz'
if not os.path.exists(options['input_path']+'wordembeddings300/'):
	os.makedirs(options['input_path']+'wordembeddings300/')
if not os.path.exists(options['input_path']+'wordembeddings100/'):
	os.makedirs(options['input_path']+'wordembeddings100/')

eps = 0.01

vecs_all = []
orig_dists = []
num_dists = []
d = 0
for (name,dataset) in zip(vec_name,datasets):
	dp = in_path.format(dataset)
	ws = loadmat(dp)
	vecs_all.append(ws[name])
	# save 300 dim version
	if dataset != 'sherlock':
		np.savez_compressed(out_path.format(300,dataset),text=ws[name])
	else:
		np.savez_compressed(out_path.format(300,dataset),text=ws[name][:,:-3])		
	# Calculate original distance matrix for cosine distances
	orig_dists.append(pairwise_distances(vecs_all[d].T, metric='cosine'))
	num_dists.append(float(len(np.ravel(orig_dists[d]))))
	d += 1

# # project to 100 dims
# l1_err_in_cosdist = 1
# while(l1_err_in_cosdist > 0.14): # empirically found to be close to the lower bound
# 	transformer100 = GaussianRandomProjection(n_components=100, eps=eps)
# 	D_100 = []
# 	dists_100dim = []
# 	l1_err_in_cosdist = 0.	
# 	for d in range(ndata):
# 		D_100.append(transformer100.fit_transform(vecs_all[d].T))
# 		D_100[d] = D_100[d].T
# 		# calculate new dists
# 		dists_100dim.append(pairwise_distances(D_100[d].T, metric='cosine'))
# 		l1_err_in_cosdist += norm(np.ravel(orig_dists[d]) - np.ravel(dists_100dim[d]), 1)/num_dists[d]

# # save results
# for d, dataset in enumerate(datasets):
# 	if dataset != 'sherlock':
# 		np.savez_compressed(out_path.format(100,dataset),text=D_100[d])
# 	else:
# 		np.savez_compressed(out_path.format(100,dataset),text=D_100[d][:,:-3])



# # preprocess sherlock
# dp1 = options['raw_path']+'wordembeddings/map_text_1st.npz'
# dp2 = options['raw_path']+'wordembeddings/map_text_2nd.npz'
# ws = np.load(dp1)
# vecs1 = ws['text']
# ws = np.load(dp2)
# vecs2 = ws['text']
# vecs = np.concatenate((vecs1,vecs2),axis=1)
# np.savez_compressed(out_path.format('sherlock'),text=vecs)













