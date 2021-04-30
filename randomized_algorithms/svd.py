import numpy as np
import scipy.linalg
from abc import ABC, abstractmethod

def DirectSVD(A, Q):
    # Form the (k+p) x n matrix
    B = Q.conj().T @ A

    # Form the SVD of the small matrix
    Uhat, d, vh = np.linalg.svd(B, full_matrices=False)

    u = Q @ Uhat

    return u, d, vh


def InterpolatoryDecomposition_row(A, k, overwrite_a=False):
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


def DirectEigenvalueDecomposition(A, Q):
    # Only if A is self adjoint

    assert np.allclose(A, A.conj().T)

    B = Q.conj().T @ A @ Q

    assert np.allclose(B, B.conj().T)
    w, v = np.linalg.eigh(B)

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


def SinglePassEigenvalueDecomposition(G, Q, Y):
    # Only if A is self adjoint

    # Solve least squares problem C (Q*G) = Q*Y for C
    #
    # Attack everything with conjugate transpose:
    # (G*Q) C* = Y*Q

    A = G.conj().T @ Q
    B = Y.conj().T @ Q

    Ch, res, _, _ = scipy.linalg.lstsq(A, B, overwrite_a=True, overwrite_b=True)

    assert np.allclose(Ch, Ch.conj().T)

    w, v = np.linalg.eigh(Ch)

    return Q @ v, w


if __name__ == "__main__":
    from range_finder import RandomizedRangeFinder

    m = 1024
    n = 512
    k = 256
    A = np.random.randn(m, n)
    A = A @ A.conj().T

    Q, G, Y = RandomizedRangeFinder(A, k, p=50, return_random_and_sample=True, svd=False)

    V, D = SinglePassEigenvalueDecomposition(G, Q, Y)

    print (np.linalg.norm(V @ np.diagflat(D) @ V.conj().T - A))


