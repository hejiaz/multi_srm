#!/usr/bin/env python
# Convert annotation of datasets into ndim embeddings using wiki embeddings and yingyu's weighting
# By Hejia Zhang @ Princeton

import numpy as np
import yaml
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics.pairwise import pairwise_distances
import os 
from numpy.linalg import norm

# final dimension
ndim = 100
wiki = True
pre = ''

# load path
setting = open('../setting.yaml')
options = yaml.safe_load(setting)

freq_file = options['raw_path']+'wordembeddings/wiki_embeddings/enwiki_vocab.txt'
if wiki:
	vecs_file = options['raw_path']+'wordembeddings/wiki_embeddings/wiki_vocab_sq_vectors_300dim.npz'
else:
	vecs_file = options['raw_path']+'wordembeddings/glove.840B.300d.txt'

# datasets 
dataset = ['GreenEyes','milky','vodka','sherlock']
antn_file = options['raw_path']+'wordembeddings/wiki_embeddings/{}_annotation.txt'
out_path = options['input_path']+pre+'wordembeddings'+str(ndim)+'/'
if not os.path.exists(out_path):
       os.makedirs(out_path)  
out_file = out_path+'{}.npz'

# get word frequency from Wiki
freq_dic = {}
with open(freq_file) as fid:
	freq = fid.readlines()
freq = [x.strip().split() for x in freq]
total_num = 0 
for i in range(len(freq)):
	freq_dic[freq[i][0]] = int(freq[i][1])
	total_num += int(freq[i][1])
for i in range(len(freq)):
	freq_dic[freq[i][0]] /= total_num

# find all words that appear in our datasets
all_words = []
for ds in dataset:
	with open(antn_file.format(ds)) as fid:
		ds_words = fid.readlines()
	ds_words = [x.strip().lstrip('\ufeff').split() for x in ds_words]
	for i in range(len(ds_words)):
		all_words.extend(ds_words[i])
all_words=list(set(all_words)) #remove duplicates

# only retain words in datasets
ds_freq = {}
for word in all_words:
	ds_freq[word] = freq_dic[word]
del freq_dic

# convert all embeddings into a dictionary: wiki
if wiki:
	vecs_dic = {}
	ws = np.load(vecs_file)
	vocab = ws['vocab']
	vectors = ws['vectors']
	vectors_dim = vectors.shape[0] # 300
	for i in range(len(vocab)):
		vecs_dic[vocab[i,0][0]] = vectors[:,i].astype(np.float32)
	# only retain words in datasets
	ds_vecs = {}
	for word in all_words:
		ds_vecs[word] = vecs_dic[word]
	del vecs_dic
# glove
else:
	vectors_dim = 300
	with open(vecs_file) as fid:
		vecs = fid.readlines()
	vecs = [x.strip() for x in vecs]
	retain = []
	for i in range(len(vecs)):
		tmp = vecs[i].split(' ',1)
		if tmp[0] in all_words:
			retain.append(i)
	ds_vecs = {}
	for line in retain:
		tmp = vecs[line].split()
		embedding = np.array(list(map(np.float32,tmp[1:])),dtype=np.float32)
		ds_vecs[tmp[0]] = embedding
	del vecs

# project embeddings to ndim
map_back = []
all_vecs = np.empty((vectors_dim,0),dtype=np.float32)
for word in ds_vecs.items():
	all_vecs = np.concatenate((all_vecs,word[1].reshape(vectors_dim,1)),axis=1)
	map_back.append(word[0])
# projection
eps = 0.01
orig_dists = pairwise_distances(all_vecs.T, metric='cosine')
num_dists = float(len(np.ravel(orig_dists)))
l1_err_in_cosdist = 1
while(l1_err_in_cosdist > 0.07): # empirically found to be close to the lower bound
	transformer100 = GaussianRandomProjection(n_components=ndim, eps=eps)
	D_100 = transformer100.fit_transform(all_vecs.T)
	D_100 = D_100.T
	# calculate new dists
	dists_100dim = pairwise_distances(D_100.T, metric='cosine')
	l1_err_in_cosdist = norm(np.ravel(orig_dists) - np.ravel(dists_100dim), 1)/num_dists
# map back
for i in range(len(map_back)):
	ds_vecs[map_back[i]] = D_100[:,i]

# for each dataset, use wiki embedding and yingyu coefficient to convert
alpha = 0.0001
for ds in dataset:
	with open(antn_file.format(ds)) as fid:
		ds_words = fid.readlines()
	ds_words = [x.strip().lstrip('\ufeff').split() for x in ds_words]
	num_words_in_TR = []
	for i in range(len(ds_words)):
		num_words_in_TR.append(len(ds_words[i]))
	text = np.zeros((ndim,len(ds_words)),dtype=np.float32)
	for i in range(len(ds_words)):
		for j in range(len(ds_words[i])):
			word = ds_words[i][j]
			factor = alpha/(ds_freq[word] + alpha)
			text[:,i] += ds_vecs[word]*factor/num_words_in_TR[i]
	if ds != 'sherlock':
		np.savez_compressed(out_file.format(ds),text=text)
	else:
		np.savez_compressed(out_file.format(ds),text=text[:,:-3])	

