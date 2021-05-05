import numpy as np
import scipy.linalg
from abc import ABC, abstractmethod

def DirectSVD(A, Q, check_finite=True, eigh=False):
    # Form the (k+p) x n matrix
    B = Q.conj().T @ A

    if eigh:
        T = B @ B.conj().T

        d, Uhat = np.linalg.eigh(T)

        d = np.sqrt(d)
        u = Q @ Uhat
        vh = np.dot(1/d * Uhat.conj().T, B)
        #vh = np.linalg.inv(np.diagflat(d)) @ Uhat.conj().T @ B
    else:
        # Form the SVD of the small matrix
        Uhat, d, vh = scipy.linalg.svd(B, full_matrices=False, overwrite_a=True,
                                       check_finite=check_finite)

        u = Q @ Uhat

    return u, d, vh

def GPUDirectSVD(A, Q, eigh=False):
    import cupy as cp
    B = Q.conj().T @ A

    if eigh:
        T = B @ B.conj().T

        d, Uhat = cp.linalg.eigh(T)

        d = np.sqrt(d)
        u = Q @ Uhat
        vh = cp.dot(1/d * Uhat.conj().T, B)
    else:
        # Form the SVD of the small matrix
        Uhat, d, vh = cp.linalg.svd(B, full_matrices=False)

        u = Q @ Uhat

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


def DirectEigenvalueDecomposition(A, Q, debug=True):
    # Only if A is self adjoint

    if debug:
        assert np.allclose(A, A.conj().T)

    B = Q.conj().T @ A @ Q

    if debug:
        assert np.allclose(B, B.conj().T)

    w, v = scipy.linalg.eigh(B, overwrite_a=True)

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


def SinglePassEigenvalueDecomposition(G, Q, Y, debug=True):
    # Only if A is self adjoint

    # Solve least squares problem C (Q*G) = Q*Y for C
    #
    # Attack everything with conjugate transpose:
    # (G*Q) C* = Y*Q

    A = G.conj().T @ Q
    B = Y.conj().T @ Q

    Ch, res, _, _ = scipy.linalg.lstsq(A, B, overwrite_a=True, overwrite_b=True)

    if debug:
        assert np.allclose(Ch, Ch.conj().T)

    w, v = np.linalg.eigh(Ch)

    return Q @ v, w


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


