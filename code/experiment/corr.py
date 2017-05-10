# mystery segment identification experiment code

import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import utils as ut

# mainly used in left-dataset-out experiment, output accuracy of each subject
# arguments:
# transformed_data: a length-2 list of 3d arrays (nfeature x time x # subjects). The two 3d arrays must have the same size
# return:
# corr: a single number between -1 and 1, Pearson correlation
def predict(transformed_data):
    ndim, nsample, nsubjs = transformed_data[0].shape
    transformed_avg = []
    for i in range(2):
        avg_data = np.zeros((ndim*nsample,1),order='f',dtype=np.float32)
        for m in range(nsubjs):
            avg_data += transformed_data[i][:,:,m].reshape(ndim*nsample,1)
        transformed_avg.append(avg_data/nsubjs)
  
    # compute correlation
    corr = ut.compute_correlation(transformed_avg[0].T,transformed_avg[1].T)

    return corr.item()


