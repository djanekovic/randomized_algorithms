import numpy as np
import scipy.linalg
from time import time

from random import sample
from abc import ABC, abstractmethod


#{{{ RandomizedRangeFinders
class BaseRangeFinder:
    def __init__(self, A, k=0, p=10, q=0, check_finite=True, debug=True):
        m, n = A.shape
        self._duration = {}

        # if k is set leave it as is, else set to min(m, n)
        k = k if k else min(m, n)
        self._l = min(k + p, min(m, n))

        # form n x l Gaussian random matrix G
        self._G = np.random.normal(size=(n, self._l))

        self._Q = self._compute(A, q, debug=debug)

    @property
    def G(self):
        return self._G

    @property
    def Y(self):
        return self._Y

    @property
    def Q(self):
        return self._Q

    @property
    def duration(self):
        return self._duration


class RandomizedRangeFinder(BaseRangeFinder):
    def __init__(self, A, k=0, p=10, q=0, check_finite=True, debug=True):
        super().__init__(A, k, p, q)

    def _compute(self, A, q, **kwargs):

        start = time()

        # Form the sample matrix m x l
        self._Y = A @ self._G
        for _ in range(q):
            Z = A.T @ self._Y
            self._Y = A @ Z
        self._duration["gemm"] = time() - start

        start = time()

        # Q is mxl
        Q, _ = np.linalg.qr(self._Y)
        self._duration["qr"] = time() - start

        return Q


#TODO: move to the GPU branch
def GPURandomizedRangeFinder(A, k=0, p=10, q=0, debug=True):
    import cupy as cp
    m, n = A.shape

    # if k is set leave it as is, else set to min(m, n)
    k = k if k else min(m, n)
    l = min(k + p, min(m, n))

    # form n x l Gaussian random matrix G
    G = cp.random.normal(size=(n, l))

    # Form the sample matrix m x l
    Y = A @ G
    for _ in range(q):
        Z = A.T @ Y
        Y = A @ Z

    # Orhonormalize the columns
    # Q is mxl
    Q, _ = cp.linalg.qr(Y, mode='reduced')

    return Q

#}}}



#{{{ RandomizedSubspaceIteration

class RandomizedSubspaceIteration(BaseRangeFinder):
    def __init__(self, A, k=0, p=10, q=0, check_finite=True, debug=True):
        super().__init__(A, k, p, q)

    def _compute(self, A, q, **kwargs):
        start = time()

        # Orhonormalize the columns
        self._Y = A @ self._G
        Q, _ = np.linalg.qr(self._Y, 'reduced')

        for _ in range(q):
            W, _ = np.linalg.qr(A.conj().T @ Q, 'reduced')
            Q, _ = np.linalg.qr(A @ W, 'reduced')
        self._duration['gemm'] = time() - start

        self._duration['qr'] = 0

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

#TODO: can we think of some magic to treat this as RangeFinder
class FastRandomizedRangeFinder():
    def __init__(self, A, k=0, p=10, q=0, check_finite=True, debug=True):
        assert q is 0
        self._duration = {}
        m, n = A.shape

        # if k is set leave it as is, else set to min(m, n)
        k = k if k else min(m, n)
        l = min(k + p, min(m, n))

        self._R = sample(list(range(n)), l)
        self._D = np.sqrt(n/l) * np.exp(2 * 1j * np.random.random_sample(size=n) * np.pi)

        start = time()
        self._Y = self._compute(A, debug)
        self._duration['gemm'] = time() - start

        start = time()
        self._Q, _ = np.linalg.qr(self._Y)
        self._duration['qr'] = time() - start

    def _compute(self, A, debug):
        Yt = np.fft.fft(np.multiply(self._D[:, None], A.T), norm='ortho', axis=0)[self._R, :]

        if debug:
            _, n = A.shape
            Y_ = A  @ np.diagflat(self._D)
            Y_ = Y_ @ scipy.linalg.dft(n, scale='sqrtn')
            Y_ = Y_ @ np.eye(n)[:, self._R]
            assert np.allclose(Y_, Yt.T)

        return Yt.T

    @property
    def Y(self):
        return self._Y

    @property
    def G(self):
        n = len(self._D)
        return np.diagflat(self._D) @ scipy.linalg.dft(n, scale='sqrtn') @ np.eye(n)[:, self._R]

    @property
    def Q(self):
        return self._Q

    @property
    def duration(self):
        return self._duration


def GPUFastRandomizedRangeFinder(A, k=0, p=20, debug=True):
    import cupy as cp
    m, n = A.shape

    # if k is set leave it as is, else set to min(m, n)
    k = k if k else min(m, n)
    l = min(k + p, min(m, n))

    R = sample(list(range(n)), l)
    D = cp.exp(2 * 1j * cp.random.sample(n) * np.pi)

    # Y.T is lxm => Y is mxl
    Yt = cp.sqrt(n/l) * cp.fft.fft(cp.multiply(D[:, None], A.T), norm='ortho', axis=0)[R, :]

    return Yt.T

#}}}

#{{{ BlockRandomizedRangeFinder

class BlockRandomizedRangeFinder(BaseRangeFinder):
    def __init__(self, A, k=0, p=10, q=0, col_num=2, check_finite=True, debug=True):
        self._col_num = col_num

        super().__init__(A, k, p, q)

    def _compute(self, A, q, debug=True):
        m, n = A.shape

        start = time()

        self._Y = np.zeros((m, self._l))
        for current_column in range(0, n, self._col_num):
            a = A[:, current_column:current_column + self._col_num]
            g = self._G[current_column:current_column + self._col_num]

            y = a @ g

            if q == 0:
                self._Y += y

            for _ in range(q):
                self._Y += np.outer(a, a) @ y
        self._duration['gemm'] = time() - start

        # Here we have all the columns and Y should be A @ G
        if debug:
            assert np.allclose(self._Y, A @ self._G)

        start = time()
        Q, _ = np.linalg.qr(self._Y)
        self._duration['qr'] = time() - start

        return Q

def GPUBlockRandomizedRangeFinder(A, k=0, p=10, q=0, col_num=10, debug=True):
    import cupy as cp
    m, n = A.shape

    # if k is set leave it as is, else set to min(m, n)
    k = k if k else min(m, n)
    l = min(k + p, min(m, n))

    # form n x (k+p) Gaussian random matrix G
    if debug:
        G = np.random.normal(size=(n, l))

    # number of columns we are pushing in the pipeline
    current_column = 0

    Y = cp.zeros((m, l))

    for current_column in range(0, n, col_num):
        a = cp.array(A[:, current_column:current_column + col_num])
        if debug:
            g = cp.array(G[current_column:current_column + col_num])
        else:
            g = cp.random.normal(size=a.shape)

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
