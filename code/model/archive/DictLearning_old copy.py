#!/usr/bin/env python

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
    U_ : 2D array, shape=[samples, factors]
        The loadings :math:`U` shared across subjects. Assume temporally sychronized stimulus
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

        # Prepare the laplacian operator for this data
        self.L_ = self._create_laplacian_operator(R)
        self.max_eigval_L_ = \
            scipy.sparse.linalg.svds(self.L_, k=1, which='LM',
                                     return_singular_vectors=False)
        self.max_eigval_L_ = (self.max_eigval_L_[0]**2)

        if self.method == 'tvl1':
            self._build_3d_v(R)
            self.weight = 2*self.mu/self.lam        

        self.U_, self.Vs_, self.V_ = self._msdl(X) # Run MSDL

        return self

    # not used
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

        # # Check if the model exist
        # if hasattr(self, 'Vs_') is False:
        #     raise NotFittedError("The model fit has not been run yet.")

        # # Check the number of subjects
        # if len(X) != len(self.Vs_):
        #     raise ValueError("The number of subjects does not match the one"
        #                      " in the model.")

        Us = [None] * len(X)
        Vu, Vsig, Vv = np.linalg.svd(self.V_, full_matrices=False)
        for subject in range(len(X)):
            U = fast_dot(fast_dot(X[subject].T,Vu),np.diag(Vsig / fast_dot(Vsig**2,Vv)))
            Us[subject] = self._update_u(X[subject], self.Vs_[subject], U)

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



    @staticmethod
    def _create_laplacian_operator(R):
        """ Pre-computes the 3D-Laplacian operator
        Parameters
        ----------
        R : list of 2D arrays, element i has shape=[voxels, 3]
            Each row in the list contains the scanner coordinate of each voxel
            of fMRI data of all subjects.
        Returns
        -------
        2D sparse array, shape=[voxels, voxels]
            The laplacian matrix.
        """
        nmax = R.max(axis=0)+1
        voxels = R.shape[0]
        cube = -np.ones((nmax),dtype=np.float32)
        cube[R[:, 0], R[:, 1], R[:, 2]] = np.arange(voxels,dtype=np.float32)

        data = np.zeros(7*voxels,dtype=np.float32)
        row_ind = np.zeros(7*voxels,dtype=np.float32)
        col_ind = np.zeros(7*voxels,dtype=np.float32)
        kernel = np.array([[-1, 0, 0], [1, 0, 0],
                           [0, -1, 0], [0, 1, 0],
                           [0, 0, -1], [0, 0, 1]],dtype=np.int32)
        total_values = 0
        for v in range(voxels):
            offset = 0
            positions = kernel + R[v, :]
            for i in range(positions.shape[0]):
                if np.any(positions[i, :] < 0) or \
                   np.any(positions[i, :] >= nmax) or \
                   cube[positions[i, 0], positions[i, 1], positions[i, 2]] < 0:
                    continue
                data[total_values+offset] = -1.0
                row_ind[total_values+offset] = v
                col_ind[total_values+offset] = cube[positions[i, 0],
                                                    positions[i, 1],
                                                    positions[i, 2]]
                offset += 1

            data[total_values+offset] = offset
            row_ind[total_values+offset] = v
            col_ind[total_values+offset] = v
            total_values += offset+1

        L = scipy.sparse.csr_matrix((data[:total_values],
                                     (row_ind[:total_values],
                                      col_ind[:total_values])),
                                    shape=(voxels, voxels),dtype=np.float32)
        return L

    def _laplacian(self, x):
        """Computes the inner product with a 3D-laplacian operator.
        Parameters
        ----------
        x : array, shape=[voxels, ] or [voxels, n]
            An array with one or more volumes represented as a column
            vectorized set of voxels.
        Returns
        -------
        array (same shape as x) with the result of applying the 3D laplacian
        operator on each column of x.
        """
        return self.L_.dot(x)

    def _objective_function(self, data, U, Vs, V):
        """Calculate the objective function of MSDL
        Parameters
        ----------
        data : list of 2D arrays, element i has shape=[voxels, samples]
            Each element in the list contains the fMRI data of one subject.
        U  : 2D array, shape=[samples, factors]
            The loadings :math:`U` shared across subjects.
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
            objective += np.linalg.norm(data[s].T - fast_dot(U,Vs[s].T),
                                        'fro')**2 \
                         + self.mu * np.linalg.norm(Vs[s] - V, 'fro')**2
        objective /= 2
        # objective += self.lam * np.sum(np.abs(V))
        objective += self.lam * (np.sum(np.abs(V)) + 0.5 *
                                 np.sum(V * self._laplacian(V)))
        return objective

    @staticmethod
    def _update_u(data, Vs, U):
        """Dictionary (loadings) update for a subject
        Parameters
        ----------
        data : a list of 2D array, shape=[voxels, samples]
            The fMRI data of all subjects.
        Vs : a list of 2D array, shape=[voxels, factors]
            The spatial map :math:`V_s` for all subjects.
        U : array, shape=[samples, factors]
            The current loadings :math:`U` 
        Returns
        -------
        U : array of shape=[samples, factors]
            The updated loadings :math:`U` 
        """
        factors = Vs[0].shape[1]
        nsubjs = len(data)
        # initialize
        A = np.zeros((factors,factors),dtype=np.float32)
        B = np.zeros(U.shape,dtype=np.float32)
        for m in range(nsubjs):
            A += fast_dot(Vs[m].T,Vs[m])
            B += fast_dot(data[m].T,Vs[m])
        for l in range(factors):
            dir = U[:, l] + np.nan_to_num((B[:, l] - fast_dot(U,A[:, l])) / A[l, l])
            U[:, l] = dir / np.amax((np.linalg.norm(dir, ord=2), 1.0))
        return U

    def _update_vs(self, data, V, U):
        """ Spatial map update
        Parameters
        ----------
        data : array, shape=[voxels, samples]
            The fMRI data of subject s.
        V : array, shape=[voxels, factors]
            The spatial map template :math:`V`.
        U : array, shape=[samples, factors]
            The current loadings :math:`U` for all subjects.
        Returns
        -------
        Vsi : array of shape=[voxels, factors]
            The updated spatial map :math:`V_s` for subject s.
        """
        factors = self.factors
        A = fast_dot(U.T,U) + self.mu * np.eye(factors,dtype=np.float32)
        Vsi = V + np.nan_to_num(np.linalg.solve(A, fast_dot(U.T,(data.T - fast_dot(U,V.T)))).T)
        return Vsi

    @staticmethod
    def _shrink(v, offset):
        """Computes soft shrinkage on the elements of an array
        Parameters
        ----------
        v : array
            An array with input values
        offset : float
            Offset for applying the shrinkage function
        Returns
        -------
        The array after applying the element-wise soft-thresholding function.
        """
        return np.sign(v) * np.max(np.abs(v) - offset, 0)

    def _prox(self, v):
        """Computes the proximal operator of a set of vectors
        Parameters
        ----------
        v : array, shape=[voxels, ]  or [voxels, n]
            One or more column-vectors containing each a volume
        Returns
        -------
        v_star : (shape=same as input) with the proximal operator applied to
            the input vectors in v
        """
        v_star = v.copy()
        z = v_star
        tau = 1.0
        kappa = self.kappa/(1 + self.gamma * self.max_eigval_L_)
        obj_fun_prev = 0
        for l in range(self.fista_iter):
            v0 = v_star
            v_star = self._shrink(z - kappa * (z - v + self.gamma *
                                               self._laplacian(z)),
                                  kappa * self.gamma)

            # Check for convergence
            obj_fun = \
                np.sum((v_star-v)**2, 0) + \
                self.gamma * (np.sum(np.abs(v_star), 0) +
                              0.5 * np.sum(v * self._laplacian(v), 0))
            if l > 0 and obj_fun > obj_fun_prev:
                return v0
            obj_fun_prev = obj_fun

            tau0 = tau
            tau = (1.0 + np.sqrt(1.0 + 4.0*tau0*tau0))/2.0
            z = v_star + (tau0-1)/tau*(v_star - v0)
        return v_star

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
        meanVs = np.zeros(Vs[0].shape,dtype=np.float32)
        for s in range(subjects):
            meanVs += Vs[s]
        meanVs /= subjects
        V = np.zeros(Vs[0].shape,dtype=np.float32)
        for l in range(Vs[0].shape[1]):
            if self.method == 'tvl1':
                V[:, l] = self._tvl1(meanVs[:, l])
            elif self.method == 'l1':
                V[:, l] = self._lasso(meanVs[:, l])
            else:
                raise Exception('invalid method')
            # V[:, l] = self._prox(meanVs[:, l])
        return V


    def _msdl(self, data):
        """Expectation-Maximization algorithm for fitting the probabilistic SRM.
        Parameters
        ----------
        data : list of 2D arrays, element i has shape=[voxels_i, samples]
            Each element in the list contains the fMRI data of one subject.
        Returns
        -------
        U : 2D array, shape=[samples, factors]
            The loadings :math:`U` shared across subjects.
        Vs : list of 2D arrays, element i has shape=[voxels, factors]
            The spatial maps :math:`V_s` for each subject.
        V : 2D array, shape=[voxels, factors]
            The spatial map template :math:`V`.
        """
        subjects = len(data)
        samples = data[0].shape[1]
        np.random.seed(self.rand_seed)
        
        # Initialization step: initialize the outputs.
        Vs = [None] * subjects
        U = np.zeros((samples,self.factors),dtype=np.float32)

        V = self._init_template(data, self.factors)

        Vu, Vsig, Vv = np.linalg.svd(V, full_matrices=False)
        for i in range(subjects):
            Vs[i] = V.copy()
            U += fast_dot(fast_dot(data[i].T,Vu),np.diag(np.nan_to_num(Vsig / fast_dot(Vsig**2,Vv))))/subjects

        print (self._objective_function(data, U, Vs, V))

        # Main loop of the algorithm
        for iteration in range(self.n_iter):

            self.gamma = self.lam/self.mu/subjects

            # Update loadings
            U = self._update_u(data, Vs, U)  

            # print ('Uiter'+str(iteration)+': '+str(self._objective_function(data, U, Vs, V)))
            # Update each subject's decomposition:
            for i in range(subjects):
                Vs[i] = self._update_vs(data[i], V, U)

            # print ('Vsiter'+str(iteration)+': '+str(self._objective_function(data, U, Vs, V)))
            # Update the spatial maps template:
            V = self._update_v(data, Vs)
            print ('Viter'+str(iteration)+': '+str(self._objective_function(data, U, Vs, V)))


        return U, Vs, V

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

        data_stacked = np.empty((voxels, samples),dtype=np.float32)
        samples = 0
        for i in range(subjects):
            data_stacked[:, samples:(samples+data[i].shape[1])] = data[i]
            samples += data[i].shape[1]

        V = np.nan_to_num(fica.fit_transform(data_stacked))

        return V