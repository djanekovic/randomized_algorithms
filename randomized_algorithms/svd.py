import numpy as np
import scipy.linalg
from time import time
from abc import ABC, abstractmethod

class BaseRSVD:
    def __init__(self, Q, **kwargs):
        self._duration = {}
        self._U, self._D, self._Vh = self._compute(Q, **kwargs)

    @property
    def U(self):
        return self._U

    @property
    def D(self):
        return self._D

    @property
    def Vh(self):
        return self._Vh

    @property
    def duration(self):
        return self._duration


class DirectSVD(BaseRSVD):
    def __init__(self, Q, **kwargs):
        super().__init__(Q, **kwargs)

    def _compute(self, Q, **kwargs):
        A = kwargs['A']
        eigh = kwargs['eigh']

        start = time()
        # Form the (k+p) x n matrix
        B = Q.conj().T @ A

        if eigh:
            T = B @ B.conj().T

            Dhat, Uhat = np.linalg.eigh(T)

            d = np.sqrt(Dhat)
            u = Q @ Uhat
            # Why this does not work and why would it be slower?!
            #vh = np.dot(np.reciprocal(d) * Uhat.conj().T, B)
            vh = np.linalg.inv(np.diagflat(d)) @ Uhat.conj().T @ B
        else:
            # Form the SVD of the small matrix
            Uhat, d, vh = np.linalg.svd(B, full_matrices=False)

            u = Q @ Uhat

        self._duration['factorization'] = time() - start

        return u, d, vh


def GPUDirectSVD(A, Q_gpu, eigh=False):
    # once tested merge this with the CPU function using: cp.get_array_module
    import cupy as cp

    B = Q_gpu.conj().T @ cp.asarray(A)


    if eigh:
        T = B @ B.conj().T

        Dhat, Uhat = cp.linalg.eigh(T)

        d = cp.sqrt(Dhat)
        u = Q_gpu @ Uhat
        vh = cp.linalg.inv(cp.diagflat(d)) @ Uhat.conj().T @ B
    else:
        # Form the SVD of the small matrix
        Uhat, d, vh = cp.linalg.svd(B, full_matrices=False)

        u = Q_gpu @ Uhat

    return u, d, vh


def InterpolatoryDecomposition_row(A, k, overwrite_a=False, debug=True):
    m, n = A.shape

    Q, R, P = scipy.linalg.qr(A, pivoting=True, overwrite_a=overwrite_a, mode='economic')
    T = np.linalg.inv(R[:k, :k]) @ R[:k, k:]
    X = np.zeros((m, k))
    X[P, :] = np.hstack((np.eye(k), T)).conj().T
    Is = P[:k]

    return X, Is

def RowExtraction(A, Y, k, p=10):
    m, n = A.shape
    k = k if k else min(m, n)

    X, Is = InterpolatoryDecomposition_row(A, k+p)
    Q, R = np.linalg.qr(X)
    F = R @ A[Is, :]

    Uhat, D, Vh = np.linalg.svd(F, full_matrices=False)

    return Q @ Uhat, D, Vh


class DirectEigenvalueDecomposition(BaseRSVD):
    def __init__(self, Q, **kwargs):
        super().__init__(Q, **kwargs)

    def _compute(self, Q, **kwargs):
        A = kwargs['A']
        debug = kwargs['debug']

        if debug:
            assert np.allclose(A, A.conj().T)

        B = Q.conj().T @ A @ Q

        if debug:
            assert np.allclose(B, B.conj().T)

        w, v = np.linalg.eigh(B)
        U = Q @ v

        return U, w, U.conj().T


def GPUDirectEigenvalueDecomposition(A, Q, debug=True):
    import cupy as cp
    if debug:
        assert cp.allclose(A, A.conj().T)

    B = Q.conj().T @ A @ Q

    if debug:
        assert cp.allclose(B, B.conj().T)

    w, v = cp.linalg.eigh(B)

    U = Q @ v

    return U, w, U.conj().T


def NystromMethod(A, Q):
    # Only if A is PSD

    B1 = A @ Q
    B2 = Q.conj().T @ B1

    L = np.linalg.cholesky(B2)

    #TODO: broken, we should just run backsubstitution
    #F = scipy.linalg.solve(L, B1, lower=True, transposed=True)
    F = B1 @ np.linalg.inv(L)

    U, D, Vh = np.linalg.svd(F, full_matrices=False)

    return U, np.power(D, 2), Vh


class SinglePassEigenvalueDecomposition(BaseRSVD):
    def __init__(self, Q, **kwargs):
        super().__init__(Q, **kwargs)

    def _compute(self, Q, **kwargs):
        G = kwargs['G']
        Y = kwargs['Y']
        debug = kwargs['debug']

        # Only if A is self adjoint

        # Solve least squares problem C (Q*G) = Q*Y for C
        #
        # Attack everything with conjugate transpose:
        # (G*Q) C* = Y*Q

        A = G.conj().T @ Q
        B = Y.conj().T @ Q

        Ch, _, _, _ = np.linalg.lstsq(A, B)

        if debug:
            print (np.linalg.norm(Ch - Ch.conj().T))
            assert np.allclose(Ch, Ch.conj().T)

        w, v = np.linalg.eigh(Ch)
        U = Q @ v
        D = w

        return U, D, U.conj().T


def GPUSinglePassEigenvalueDecomposition(G, Q, Y, debug=True):
    import cupy as cp
    A = G.conj().T @ Q
    B = Y.conj().T @ Q

    Ch, res, _, _ = cp.linalg.lstsq(A, B)

    if debug:
        assert cp.allclose(Ch, Ch.conj().T)

    w, v = cp.linalg.eigh(Ch)

    return Q @ v, w


if __name__ == "__main__":
    from range_finder import RandomizedRangeFinder

    m = 1024
    n = 512
    k = 256
    A = np.random.randn(m, n)

    Q = RandomizedRangeFinder(A, k=k)
    U, D, Vh = DirectSVD(A, Q, eigh=True)

    print (np.linalg.norm(A - U @ np.diagflat(D) @ Vh))


