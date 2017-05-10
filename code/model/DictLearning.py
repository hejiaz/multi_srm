#  Copyright 2016 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Multi-subject Dictionary Learning
The implementation is based on the following publication:
.. [Varoquaux2011] "Multi-subject Dictionary Learning to Segment an Atlas of
   Brain Spontaneous Activity",
   G. Varoquaux, A. Gramfort, F. Pedregosa, V. Michel, .B. Thirion
   22nd International Conference Information Processing in Medical Imaging
   (IPMI 2011)
   http://rd.springer.com/chapter/10.1007/978-3-642-22092-0_46
"""

# Authors: Javier Turek (Intel Labs), 2017

import numpy as np
import scipy
import scipy.sparse
from .sklearnica import FastICA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import assert_all_finite
from sklearn.utils.validation import NotFittedError
from sklearn.utils.extmath import fast_dot
from skimage.restoration import denoise_tv_bregman
from lightning.regression import FistaRegressor

class MSDL(BaseEstimator, TransformerMixin):
    """Multi-subject Dictionary Learning
    Given multi-subject data, factorize it as spatial maps V_i and loadings
    U_i per subject:
    .. math:: X_i \\approx U_i V_i^T + E_i, \\forall i=1 \\dots N
    E_i is assumed to be white Gaussian Noise N(0,\\sigmaI)
    U_i is assumed to be Gaussian N(0,\\Sigma_U) (same covariance for all
    subjects)
    V_i = V + F_i, where F_i is Gaussian N(0,\\xsiI)
    V is a shared template across all subjects and it is assumed to have
    unit column norm
    Parameters
    ----------
    n_iter : int, default: 10
        Number of iterations to run the algorithm.
    factors : int, default: 10
        Number of factors to decompose the data.
    rand_seed : int, default: 0
        Seed for initializing the random number generator.
    mu : float, default: 1.0
        Value of the subject-global spatial map matching term.
    lam : float, default: 1.0
        Value of the regularization parameter.
    kappa : float, default: 0.5
        Step size for the FISTA algorithm. Should be a value in (0.0, 1.0).
    fista_iter : int, default: 20
        Number of iterations to run the FISTA algorithm to update the spatial
        map template.
    Attributes
    ----------
    Us_ : list of 2D arrays, element i has shape=[samples, factors]
        The loadings :math:`U_s` for each subject.
    Vs_ : list of 2D arrays, element i has shape=[voxels, factors]
        The spatial maps :math:`V_s` for each subject.
    V_ : 2D array, shape=[voxels, factors]
        The spatial map template :math:`V`.
    Note
    ----
        The number of voxels should be the same between subjects.
        The Multi-Subject Dictionary Learning is approximated using a
        Block Coordinate Descent (BCD) algorithm proposed in [Varoquaux2011]_.
        This is a single node version.
        The run-time complexity is
        :math:`O(S (V T K + (V + T) K^2 + K^3 + K I_{V}))` with
        I - the number of iterations of FISTA, V - the voxels from a subject,
        T - the number of time samples, K - the number of features/factors
        (typically, :math:`V \\gg T \\gg K`), and S - the number of subjects.
    """

    def __init__(self, n_iter=10, factors=10, rand_seed=0, mu=1.0, lam=1.0,
                 kappa=0.5, fista_iter=20, method='tvl1'):
        self.n_iter = n_iter
        self.factors = factors
        self.rand_seed = rand_seed
        self.mu = mu
        self.lam = lam
        self.fista_iter = fista_iter
        self.kappa = kappa
        self.method = method
        return

    def fit(self, X, y=None, R=None):
        """Compute the Multi-Subject Dictionary Learning decomposition
        Parameters
        ----------
        X : list of 2D arrays, element i has shape=[voxels, samples]
            Each element in the list contains the fMRI data of one subject.
        y : not used
        R : list of 2D arrays, element i has shape=[voxels, 3]
            Each row in the list contains the scanner coordinate of each voxel
            of fMRI data of all subjects.
        """

        if self.method == 'tvl1':
            self._build_3d_v(R)
            self.weight = 2*self.mu/self.lam

        # Run MSDL
        self.Us_, self.Vs_, self.V_ = self._msdl(X)

        return self

    def transform(self, X, y=None):
        """Use the model to transform data to the low-dimentional subspace
        Parameters
        ----------
        X : list of 2D arrays, element i has shape=[voxels, samples]
            Each element in the list contains the fMRI data of one subject.
        y : not used
        Returns
        -------
        Us : list of 2D arrays, element i has shape=[factors, samples]
            Loadings for each subject new given data.
        """

        Us = [None] * len(X)
        Vu, Vsig, Vv = np.linalg.svd(self.V_, full_matrices=False)
        for subject in range(len(X)):
            U = X[subject].T.dot(Vu).dot(np.diag(Vsig / (Vsig**2).dot(Vv)))
            Us[subject] = self._update_us(X[subject], self.Vs_[subject], U)

        return Us


    def _build_3d_v(self,R):
        min_idx = np.min(R,axis=0)
        R_new = R - min_idx
        max_idx = np.max(R,axis=0) - min_idx
        self.Q = (R_new[:,0],R_new[:,1],R_new[:,2])
        self.meanv = np.zeros((max_idx[0]+1,max_idx[1]+1,max_idx[2]+1),dtype=np.float32)
        self.newv = np.zeros_like(self.meanv)
        return self
 

    def _tvl1(self,v):
        mult = np.max(abs(v))
        self.meanv[self.Q[0],self.Q[1],self.Q[2]] = v/mult
        self.newv = denoise_tv_bregman(self.meanv,self.weight*mult)
        v_new = self.newv[self.Q[0],self.Q[1],self.Q[2]]
        return v_new    


    def _lasso(self,v):
        voxels = v.shape[0]
        lasso = FistaRegressor(C=self.mu,alpha=self.lam,penalty='l1')
        lasso.fit(np.eye(voxels),v)
        return lasso.coef_


    def _objective_function(self, data, Us, Vs, V):
        """Calculate the objective function of MSDL
        Parameters
        ----------
        data : list of 2D arrays, element i has shape=[voxels, samples]
            Each element in the list contains the fMRI data of one subject.
        Us : list of 2D arrays, element i has shape=[samples, factors]
            The loadings :math:`U_s` for each subject.
        Vs : list of 2D arrays, element i has shape=[voxels, factors]
            The spatial maps :math:`V_s` for each subject.
        V : 2D array, shape=[voxels, factors]
            The template spatial map :math:`V`.
        Returns
        -------
        objective : float
            The objective function value.
        """
        subjects = len(data)
        objective = 0.0
        for s in range(subjects):
            objective += np.linalg.norm(data[s].T - fast_dot(Us[s],Vs[s].T),
                                        'fro')**2 \
                         + self.mu * np.linalg.norm(Vs[s] - V, 'fro')**2
        objective /= 2
        objective += self.lam * np.sum(np.abs(V))

        return objective

    @staticmethod
    def _update_us(data, Vs, Us):
        """Dictionary (loadings) update for a subject
        Parameters
        ----------
        data : array, shape=[voxels, samples]
            The fMRI data of subject s.
        Vs : array, shape=[voxels, factors]
            The spatial map :math:`V_s` for subject s.
        Us : array, shape=[samples, factors]
            The current loadings :math:`U_s` for subject s.
        Returns
        -------
        Us : array of shape=[samples, factors]
            The updated loadings :math:`U_s` for subject s.
        """
        factors = Vs.shape[1]
        A = fast_dot(Vs.T,Vs)
        B = fast_dot(data.T,Vs)
        for l in range(factors):
            dir = Us[:, l] + np.nan_to_num((B[:, l] - fast_dot(Us,A[:, l])) / A[l, l])
            Us[:, l] = np.nan_to_num(dir / np.amax((np.linalg.norm(dir, ord=2), 1.0)))
        return Us

    def _update_vs(self, data, V, Us):
        """ Spatial map update
        Parameters
        ----------
        data : array, shape=[voxels, samples]
            The fMRI data of subject s.
        V : array, shape=[voxels, factors]
            The spatial map template :math:`V`.
        Us : array, shape=[samples, factors]
            The current loadings :math:`U_s` for subject s.
        Returns
        -------
        Vsi : array of shape=[voxels, factors]
            The updated spatial map :math:`V_s` for subject s.
        """
        factors = self.factors
        A = fast_dot(Us.T,Us) + self.mu * np.eye(factors)
        Vsi = V + np.nan_to_num(np.linalg.solve(A, fast_dot(Us.T,(data.T - fast_dot(Us,V.T)))).T)
        return Vsi

    def _update_v(self, data, Vs):
        """ Spatial map template update
        Parameters
        ----------
        data : list of 2D arrays, element i has shape=[voxels, samples]
            Each element in the list contains the fMRI data of one subject.
        Vs : list of 2D arrays, element i has shape=[voxels, factors]
            The spatial maps :math:`V_s` for each subject.
        Returns
        -------
        V : 2D array, shape=[voxels, factors]
            The updated spatial map template :math:`V`.
        """
        subjects = len(data)
        meanVs = np.zeros(Vs[0].shape)
        for s in range(subjects):
            meanVs += Vs[s]
        meanVs /= subjects
        V = np.zeros(Vs[0].shape)
        for l in range(Vs[0].shape[1]):
            if self.method == 'tvl1':
                V[:, l] = np.nan_to_num(self._tvl1(meanVs[:, l]))
            elif self.method == 'l1':
                V[:, l] = np.nan_to_num(self._lasso(meanVs[:, l]))
            else:
                raise Exception('invalid method')
        return V

    def _msdl(self, data):
        """Expectation-Maximization algorithm for fitting the probabilistic SRM.
        Parameters
        ----------
        data : list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.
        Returns
        -------
        Us : list of 2D arrays, element i has shape=[samples, factors]
            The loadings :math:`U_s` for each subject.
        Vs : list of 2D arrays, element i has shape=[voxels, factors]
            The spatial maps :math:`V_s` for each subject.
        V : 2D array, shape=[voxels, factors]
            The spatial map template :math:`V`.
        """
        subjects = len(data)
        np.random.seed(self.rand_seed)

        # Initialization step: initialize the outputs.
        Vs = [None] * subjects
        Us = [None] * subjects

        V = self._init_template(data, self.factors)

        Vu, Vsig, Vv = np.linalg.svd(V, full_matrices=False)
        for i in range(subjects):
            Vs[i] = V.copy()
            Us[i] = fast_dot(fast_dot(data[i].T,Vu),np.nan_to_num(np.diag(Vsig / fast_dot(Vsig**2,Vv))))

        # Calculate the current objective function value
        print(self._objective_function(data, Us, Vs, V))

        # Main loop of the algorithm
        for iteration in range(self.n_iter):

            # Update each subject's decomposition:
            for i in range(subjects):
                Us[i] = self._update_us(data[i], Vs[i], Us[i])
                Vs[i] = self._update_vs(data[i], V, Us[i])

            # Update the spatial maps template:
            V = self._update_v(data, Vs)
            print('After V update %d' %
                        self._objective_function(data, Us, Vs, V))

        return Us, Vs, V

    def _init_template(self, data, factors):
        """Initialize the template spatial map (V) for the MSDL with ICA.
        Parameters
        ----------
        data : list of 2D arrays, element i has shape=[voxels, samples]
            Each element in the list contains the fMRI data of one subject.
        factors : int
            The number of factors in the model.
        Returns
        -------
        V : array, shape=[voxels, factors]
            The initialized template spatial map :math:`V`.
        """
        subjects = len(data)
        voxels = data[0].shape[0]
        fica = FastICA(n_components=factors, whiten=True, max_iter=200,
                       random_state=self.rand_seed)
        samples = 0
        for i in range(subjects):
            samples += data[i].shape[1]

        data_stacked = np.empty((voxels, samples))
        samples = 0
        for i in range(subjects):
            data_stacked[:, samples:(samples+data[i].shape[1])] = data[i]
            samples += data[i].shape[1]

        V = np.nan_to_num(fica.fit_transform(data_stacked))

        return V