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

# Authors: Yida Wang
# (Intel Labs), 2016

cimport scipy.linalg.cython_blas as blas

def compute_single_matrix_multiplication(py_trans_a, py_trans_b, py_m, py_n,
                                         py_k, py_alpha, py_a, py_lda,
                                         py_b, py_ldb, py_beta, py_c, py_ldc):
    """ use blas API gemm wrapped by scipy to do matrix multiplication

    This is to compute the matrix multiplication.
    The blas APIs process matrices in column-major,
    but our matrices are in row-major, so we play the transpose trick here,
    i.e. A*B=(B^T*A^T)^T

    Parameters
    ----------
    py_trans_a: str
    do transpose or not for the first matrix A

    py_trans_b: str
    do transpose or not for the first matrix B

    py_m: int
    the row of the resulting matrix C

    py_n: int
    the column of the resulting matrix C

    py_k: int
    the collapsed dimension of the multiplying matrices
    i.e. the column of the first matrix after transpose if necessary
    the row of the second matrix after transpose if necessary

    py_alpha: float
    the weight applied to the input matrix A

    py_a: 2D array

    py_lda: int
    the stride of the input matrix A

    py_b: 2D array

    py_ldb: int
    the stride of the input matrix B

    py_beta: float
    the weight applied to the resulting matrix C

    py_c: 2D array
    in shape [py_m, py_n] of column-major
    in fact it is
    in shape [py_n, py_m] of row-major

    py_ldc: int
    the stride of the resulting matrix

    Returns
    -------
    py_c: 2D array
    in shape [py_m, py_n] of column-major
    write the resulting matrix
    """
    cdef bytes by_trans_a=py_trans_a.encode()
    cdef bytes by_trans_b=py_trans_b.encode()
    cdef char* trans_a = by_trans_a
    cdef char* trans_b = by_trans_b
    cdef int M, N, K, lda, ldb, ldc
    M = py_m
    N = py_n
    K = py_k
    lda = py_lda
    ldb = py_ldb
    ldc = py_ldc
    cdef float alpha, beta
    alpha = py_alpha
    beta = py_beta
    cdef float[:, ::1] A
    A = py_a
    cdef float[:, ::1] B
    B = py_b
    cdef float[:, ::1] C
    C = py_c
    blas.sgemm(trans_a, trans_b, &M, &N, &K, &alpha, &A[0, 0], &lda,
               &B[0, 0], &ldb, &beta, &C[0, 0], &ldc)

