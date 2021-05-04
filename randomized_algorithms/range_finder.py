import numpy as np
import scipy.linalg

from random import sample
from abc import ABC, abstractmethod


#{{{ RandomizedRangeFinders

def RandomizedRangeFinder(A, k=0, p=10, q=0, check_finite=True, svd=False,
                          return_random_and_sample=False, debug=True):
    m, n = A.shape

    # if k is set leave it as is, else set to min(m, n)
    k = k if k else min(m, n)
    l = min(k + p, min(m, n))

    # form n x l Gaussian random matrix G
    G = np.random.normal(size=(n, l))

    # Form the sample matrix m x l
    if q:
        Y = np.linalg.matrix_power(A @ A.conj().T, q) @ A @ G
    else:
        Y = A @ G

    # Orhonormalize the columns
    if svd:
        U, _, _ = np.linalg.svd(Y, full_matrices=False)
        Q = U[:, :k]
    else:
        # Q is mxl
        overwrite_a = False if return_random_and_sample else True
        Q, _ = scipy.linalg.qr(Y, mode='economic', overwrite_a=overwrite_a,
                               check_finite=check_finite)

    if return_random_and_sample:
        return Q, G, Y

    return Q


def GPURandomizedRangeFinderQR(A, k=0, p=10, q=0, debug=True):
    import cupy as cp
    m, n = A.shape

    # if k is set leave it as is, else set to min(m, n)
    k = k if k else min(m, n)
    l = min(k + p, min(m, n))

    # form n x l Gaussian random matrix G
    G = cp.random.normal(size=(n, l))

    # Form the sample matrix m x l
    if q:
        Y = cp.linalg.matrix_power(A @ A.conj().T, q) @ A @ G
    else:
        Y = A @ G

    # Orhonormalize the columns
    # Q is mxl
    Q, _ = cp.linalg.qr(Y, mode='reduced')

    return Q

#}}}



#{{{ RandomizedSubspaceIteration

def RandomizedSubspaceIteration(A, k=0, p=10, q=0, return_random_and_sample=False, debug=True):
    m, n = A.shape

    # if k is set leave it as is, else set to min(m, n)
    k = k if k else min(m, n)
    l = min(k + p, min(m, n))

    # form n x l Gaussian random matrix G
    G = np.random.normal(size=(n, l))

    # Form the sample matrix m x l
    Y = A @ G

    # Orhonormalize the columns
    Q, _ = np.linalg.qr(Y, 'reduced')

    for _ in range(q):
        W, _ = np.linalg.qr(A.conj().T @ Q, 'reduced')
        Q, _ = np.linalg.qr(A @ W, 'reduced')

    if return_random_and_sample:
        return Q, G, Y

    return Q


def GPURandomizedSubspaceIteration(A, k=0, p=10, q=0, debug=True):
    import cupy as cp
    m, n = A.shape

    # if k is set leave it as is, else set to min(m, n)
    k = k if k else min(m, n)
    l = min(k + p, min(m, n))

    # form n x l Gaussian random matrix G
    G = cp.random.normal(size=(n, l))

    # Form the sample matrix m x l
    Y = A @ G

    # Orhonormalize the columns
    Q, _ = cp.linalg.qr(Y, 'reduced')

    for _ in range(q):
        W, _ = cp.linalg.qr(A.conj().T @ Q, 'reduced')
        Q, _ = cp.linalg.qr(A @ W, 'reduced')

    return Q

#}}}

#{{{ FastRandomizedRangeFinder

def FastRandomizedRangeFinder(A, k=0, p=20, debug=True):
    m, n = A.shape

    # if k is set leave it as is, else set to min(m, n)
    k = k if k else min(m, n)
    l = min(k + p, min(m, n))

    # Compute Y = A Omega where Omega = sqrt(n/k+p) D F R.
    # - D is an nxn diagonal matrix whose entries are independent random variables uniformly
    #   distributed on the complex unit circle
    # - F is the nxn unitary DFT matrix
    # - R is an nxk+p matrix whose columns are drawn randomly without replacement from the
    #   columns of the nxn identity matrix.
    # we want to compute Y as fast as possible!
    # Since F is DFT matrix we can apply it in O(nlogn), notice D and R have O(n) nonzero
    # elements.
    #
    # Y = A Omega
    # Y.T = Omega.T A.T
    # Y.T = norm R.T F.T D.T A.T    -> F == F.T, D == D.T
    # Y.T = norm R.T F D A.T
    #
    # Since we are dealing with matrices that have O(n) nonzeros we have to smart how we
    # handle them.
    # Action of matrix R.T on some other matrix X is simple row extraction:
    #
    # |1 0 0| |a b| = |a b|
    # |0 0 1| |c d|   |e f|
    #         |e f|
    #
    #
    R = sample(list(range(n)), l)
    D = np.exp(2 * 1j * np.random.random_sample(size=n) * np.pi)

    # Y.T is lxm => Y is mxl
    Yt = np.sqrt(n/l) * np.fft.fft(np.multiply(D[:, None], A.T), norm='ortho', axis=0)[R, :]
    if debug:
        import scipy.linalg
        Y_ = np.sqrt(n/l) * A @ np.diagflat(D) @ scipy.linalg.dft(n, scale='sqrtn') @ np.eye(n)[:, R]
        assert np.allclose(Y_.T, Yt)

    return Yt.T


def GPUFastRandomizedRangeFinder(A, k=0, p=20, debug=True):
    import cupy as cp
    m, n = A.shape

    # if k is set leave it as is, else set to min(m, n)
    k = k if k else min(m, n)
    l = min(k + p, min(m, n))

    R = random.sample(list(range(n)), l)
    D = cp.exp(2 * 1j * cp.random.sample(n) * np.pi)

    # Y.T is lxm => Y is mxl
    Yt = cp.sqrt(n/l) * cp.fft.fft(cp.multiply(D[:, None], A.T), norm='ortho', axis=0)[R, :]

    return Y.T

#}}}

#{{{ BlockRandomizedRangeFinder

def BlockRandomizedRangeFinder(A, k=0, p=10, q=0, col_num=2, check_finite=True,
                               return_random_and_sample=False, debug=True):
    m, n = A.shape

    # if k is set leave it as is, else set to min(m, n)
    k = k if k else min(m, n)
    l = min(k + p, min(m, n))

    # form n x l Gaussian random matrix G
    G = np.random.normal(size=(n, l))

    # number of columns we are pushing in the pipeline
    current_column = 0

    Y = np.zeros((m, l))

    for current_column in range(0, n, col_num):
        a = A[:, current_column:current_column + col_num]
        g = G[current_column:current_column + col_num]

        y = a @ g

        if q == 0:
            Y += y

        for _ in range(q):
            Y += np.outer(a, a) @ y

    # Here we have all the columns and Y should be A @ G
    if debug:
        assert np.allclose(Y, A @ G)

    overwrite_a = False if return_random_and_sample else True
    Q, _ = scipy.linalg.qr(Y, mode='economic', overwrite_a=overwrite_a,
                           check_finite=check_finite)

    if return_random_and_sample:
        return Q, G, Y

    return Q



def GPUBlockRandomizedRangeFinder(A, k=0, p=10, q=0, col_num=10, debug=True):
    import cupy as cp
    m, n = A.shape

    # if k is set leave it as is, else set to min(m, n)
    k = k if k else min(m, n)
    l = min(k + p, min(m, n))

    # form n x (k+p) Gaussian random matrix G
    G = np.random.normal(size=(n, l))

    # number of columns we are pushing in the pipeline
    current_column = 0

    Y = cp.zeros((m, l))

    for current_column in range(0, n, col_num):
        a = cp.array(A[:, current_column:current_column + col_num])
        g = cp.array(G[current_column:current_column + col_num])

        y = a @ g

        if q == 0:
            Y += y

        for _ in range(q):
            Y += cp.outer(a, a) @ y

    # Here we have all the columns and Y should be A @ G
    if debug:
        assert cp.allclose(Y, A @ G)

    Q, _ = cp.linalg.qr(Y, mode='reduced')

    return Q

#}}}
