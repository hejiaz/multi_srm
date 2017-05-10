# image prediction experiment code
# transformed_data is a list 

import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath('..'))
from sklearn.svm import NuSVC
import warnings

# time here is number of scenes
# transformed_data: a list of 2d arrays (nfeature x time) of length (# subjects)
# label: a list of 1d arrays (time,) of length (# subjects)
# accu: a 1d array (# subjects,)
def predict(transformed_data,label):
    nsubjs = len(transformed_data)
    accu = np.zeros((nsubjs),dtype=np.float32)
  
    for tst_subj in range(nsubjs):
        tst_data = transformed_data[tst_subj]
        tst_label = label[tst_subj]

        trn_subj = list(range(nsubjs))
        trn_subj.remove(tst_subj)    
        trn_data = transformed_data[trn_subj[0]]
        trn_label = label[trn_subj[0]]

        for m in range(1,nsubjs-1):
            trn_data = np.concatenate((trn_data, transformed_data[trn_subj[m]]), axis = 1)
            trn_label = np.concatenate((trn_label, label[trn_subj[m]]))
    
        # scikit-learn svm for classification
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nu_vec=[0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0.01]
            for nu in nu_vec:
                try:
                    clf = NuSVC(nu=nu, kernel = 'linear',max_iter=1000)
                    clf.fit(trn_data.T, trn_label)
                    pred_label = clf.predict(tst_data.T)
                    accu[tst_subj] = sum(pred_label == tst_label)/len(pred_label)
                    break
                except ValueError:
                    accu[tst_subj] = 0.
    return accu

def predict_loo(transformed_data,label,loo_subj):
    # accuracy for left-out subjects
    nsubjs = len(transformed_data)
    accu = np.zeros((len(loo_subj)),dtype=np.float32)
    trn_subj = list(range(nsubjs))
    trn_subj=[n for n in trn_subj if n not in loo_subj]     

    trn_data = transformed_data[trn_subj[0]]
    trn_label = label[trn_subj[0]]

    for m in range(1,len(trn_subj)):
        trn_data = np.concatenate((trn_data, transformed_data[trn_subj[m]]), axis = 1)
        trn_label = np.concatenate((trn_label, label[trn_subj[m]]))

    for idx,tst_subj in enumerate(loo_subj):
        tst_data = transformed_data[tst_subj]
        tst_label = label[tst_subj]
        # scikit-learn svm for classification
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nu_vec=[0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0.01]
            for nu in nu_vec:
                try:
                    clf = NuSVC(nu=nu, kernel = 'linear',max_iter=1000)
                    clf.fit(trn_data.T, trn_label)
                    pred_label = clf.predict(tst_data.T)
                    accu[idx] = sum(pred_label == tst_label)/len(pred_label)
                    break
                except ValueError:
                    accu[idx] = 0.                         
    return accu
