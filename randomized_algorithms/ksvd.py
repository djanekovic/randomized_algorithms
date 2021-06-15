import numpy as np
import scipy as sp
from sklearn.linear_model import orthogonal_mp_gram
from .range_finder import RandomizedRangeFinder
from .svd import DirectSVD

from abc import ABC, abstractmethod

def rsvd(A, k, p=10, q=0):
    r = RandomizedRangeFinder(A, k=k, p=p, q=q, debug=False)
    rsvd = DirectSVD(r.Q, A=A, eigh=False)
    return rsvd.U, rsvd.D, rsvd.Vh

class AbstractKSVD(ABC):
    def __init__(self, n_components, max_iter=10, tol=1e-6,
                 transform_n_nonzero_coefs=None):
        """
        Parameters
        ----------
        n_components:
            Number of dictionary elements

        max_iter:
            Maximum number of iterations

        tol:
            tolerance for error

        transform_n_nonzero_coefs:
            Number of nonzero coefficients to target
        """
        self.components_ = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.error_norm = []

    @abstractmethod
    def update_dict(self):
        pass

    def _initialize(self, X):
        """Initialize dictionary D

        Problem is posed as find X, D such that min{Y - X D}.

        We know that X D is of rank n_components hence we find SVD of Y and return
        right singular vectors scaled with singular values.

        Note that this representation is transposed to match the sklearn representation
        of data matrix. Much more natural view of this problem is find

        Y = D X where D acts on X and not the other way around.


        Parameters
        ----------

        X: ndarray of shape (n_samples, n_features)
            Data matrix.
        """

        if min(X.shape) < self.n_components:
            D = np.random.randn(self.n_components, X.shape[1])
        else:
            u, s, vt = sp.sparse.linalg.svds(X, k=self.n_components)
            D = np.dot(np.diag(s), vt)
        # normalize each sample
        D /= np.linalg.norm(D, axis=1)[:, np.newaxis]
        return D

    def _transform(self, D, X):
        gram = D.dot(D.T)
        Xy = D.dot(X.T)

        n_nonzero_coefs = self.transform_n_nonzero_coefs
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.1 * X.shape[1])

        return orthogonal_mp_gram(gram, Xy, n_nonzero_coefs=n_nonzero_coefs).T


    def fit(self, X):
        """
        Parameters
        ----------
        X: shape = [n_samples, n_features]
        """
        D = self._initialize(X)
        for i in range(self.max_iter):
            gamma = self._transform(D, X)
            e = np.linalg.norm(X - gamma.dot(D))
            self.error_norm.append(e)
            if e < self.tol:
                print ("Number of iterations: {}", i)
                break
            D, gamma = self.update_dict(X, D, gamma)
            #assert np.all(np.isclose(np.linalg.norm(D, axis=1), 1))
        self.components_ = D
        return self

    def transform(self, X):
        return self._transform(self.components_, X)


class ClassicKSVD(AbstractKSVD):
    def __init__(self, n_components, max_iter=10, tol=1e-6,
                 transform_n_nonzero_coefs=None):
        super().__init__(n_components, max_iter, tol, transform_n_nonzero_coefs)


    def update_dict(self, X, D, gamma):
        for j in range(self.n_components):
            # index set of all the elems in the j-th row that are nonzero
            #I = gamma[:, j] > 0
            I = np.flatnonzero(gamma[:, j])
            if len(I) == 0:
                continue

            # zero out j-th row of dictionary and compute error matrix E
            # This is error without d_j
            D[j, :] = 0
            E = X[I, :] - gamma[I, :].dot(D)

            # compute rank-1 approximation of E
            u, d, vh = np.linalg.svd(E, full_matrices=False)

            D[j, :] = vh[0]
            gamma[I, j] = d[0] * u[:, 0]
        return D, gamma


class RandomizedKSVD(AbstractKSVD):
    def __init__(self, n_components, max_iter=10, tol=1e-6,
                 transform_n_nonzero_coefs=None):
        super().__init__(n_components, max_iter, tol, transform_n_nonzero_coefs)


    def update_dict(self, X, D, gamma):
        for j in range(self.n_components):
            # index set of all the elems in the j-th row that are nonzero
            #I = gamma[:, j] > 0
            I = np.flatnonzero(gamma[:, j])
            if len(I) == 0:
                continue

            # zero out j-th row of dictionary and compute error matrix E
            # This is error without d_j
            D[j, :] = 0
            E = X[I, :] - gamma[I, :].dot(D)

            # compute rank-1 approximation of E

            u, d, vh = rsvd(E, k=1, p=2, q=1)
            # Fine tuning code
            #u_, d_, vh_ = np.linalg.svd(E, full_matrices=False)

            #E_rsvd = d[0] * np.outer(u[:, 0], vh[0])
            #E_full = d_[0] * np.outer(u_[:, 0], vh_[0])

            #print (np.linalg.norm(E - E_rsvd), np.linalg.norm(E - E_full))

            D[j, :] = vh[0]
            gamma[I, j] = d[0] * u[:, 0]
        return D, gamma


class ApproximateKSVD(AbstractKSVD):
    def __init__(self, n_components, max_iter=10, tol=1e-6,
                 transform_n_nonzero_coefs=None):
        super().__init__(n_components, max_iter, tol, transform_n_nonzero_coefs)

    def update_dict(self, X, D, gamma):
        for j in range(self.n_components):
            # index map of all the atom indices in sparse representation
            #I = gamma[:, j] > 0
            I = np.flatnonzero(gamma[:, j])
            if len(I) == 0:
                continue

            D[j, :] = 0
            # approximate SVD with power method
            g = gamma[I, j].T
            r = X[I, :] - gamma[I, :].dot(D)

            d = r.T.dot(g)
            d /= np.linalg.norm(d)
            g = r.dot(d)

            # insert new column and row
            D[j, :] = d
            gamma[I, j] = g.T

        return D, gamma
