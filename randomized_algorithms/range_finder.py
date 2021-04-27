import numpy as np
from random import sample
from abc import ABC, abstractmethod

class AbstractRangeFinder(ABC):
    def __init__(self, k, p, q):
        self.k = k
        self.p = p
        self.q = q


    @abstractmethod
    def compute(self, A):
        """
        Given a mxn matrix A, this function shall compute mxl ON matrix whose
        range approximates the range of A.
        """
        pass


class RandomizedRangeFinder(AbstractRangeFinder):
    def __init__(self, k=0, p=10, q=0):
        super().__init__(k, p, q)

    def compute(self, A):
        m, n = A.shape

        # if self.k is set leave it as is, else set to min(m, n)
        self.k = self.k if self.k else min(m, n)

        # form n x (k+p) Gaussian random matrix G
        G = np.random.normal(size=(n, self.k+self.p))

        # Form the sample matrix m x (k+p)
        if self.q:
            Y = np.matrix_power(A @ A.conj().T, self.q) @ A @ G
        else:
            Y = A @ G

        # Orhonormalize the columns
        # Q is mx(k+p)
        Q, _ = np.linalg.qr(Y, mode='reduced')

        return Q


class GPURandomizedRangeFinder(AbstractRangeFinder):
    def __init__(self, k=0, p=10, q=0):
        super().__init__(k, p, q)

    def compute(self, A):
        import cupy as cp
        m, n = A.shape

        # if self.k is set leave it as is, else set to min(m, n)
        self.k = self.k if self.k else min(m, n)

        # form n x (k+p) Gaussian random matrix G
        G = cp.random.normal(size=(n, self.k+self.p))

        # Form the sample matrix m x (k+p)
        if self.q:
            Y = cp.matrix_power(A @ A.conj().T, self.q) @ A @ G
        else:
            Y = A @ G

        # Orhonormalize the columns
        # Q is mx(k+p)
        Q, _ = cp.linalg.qr(Y, mode='reduced')

        return Q


class RandomizedSubspaceIteration(AbstractRangeFinder):
    def __init__(self, k=0, p=10, q=0):
        super().__init__(k, p, q)

    def compute(self, A):
        m, n = A.shape

        # if self.k is set leave it as is, else set to min(m, n)
        self.k = self.k if self.k else min(m, n)

        # form n x (k+p) Gaussian random matrix G
        G = np.random.normal(size=(n, self.k+self.p))

        # Form the sample matrix m x (k+p)
        Y = A @ G

        # Orhonormalize the columns
        Q, _ = np.linalg.qr(Y, 'reduced')

        for _ in range(self.q):
            W, _ = np.linalg.qr(A.conj().T @ Q, 'reduced')
            Q, _ = np.linalg.qr(A @ W, 'reduced')

        return Q


class GPURandomizedSubspaceIteration(AbstractRangeFinder):
    def __init__(self, k=0, p=10, q=0):
        super().__init__(k, p, q)

    def compute(self, A):
        import cupy as cp
        m, n = A.shape

        # if self.k is set leave it as is, else set to min(m, n)
        self.k = self.k if self.k else min(m, n)

        # form n x (k+p) Gaussian random matrix G
        G = cp.random.normal(size=(n, self.k+self.p))

        # Form the sample matrix m x (k+p)
        Y = A @ G

        # Orhonormalize the columns
        Q, _ = cp.linalg.qr(Y, 'reduced')

        for _ in range(self.q):
            W, _ = cp.linalg.qr(A.conj().T @ Q, 'reduced')
            Q, _ = cp.linalg.qr(A @ W, 'reduced')

        return Q


class FastRandomizedRangeFinder(AbstractRangeFinder):
    def __init__(self, k=0, p=20):
        super().__init__(k, p, 0)

    def compute(self, A):
        m, n = A.shape

        # if self.k is set leave it as is, else set to min(m, n)
        self.k = self.k if self.k else min(m, n)

        l = self.k + self.p

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
        import scipy.linalg
        Y_ = np.sqrt(n/l) * A @ np.diagflat(D) @ scipy.linalg.dft(n, scale='sqrtn') @ np.eye(n)[:, R]
        assert np.allclose(Y_.T, Yt)

        # Q is mxl ON matrix
        Q, _ = np.linalg.qr(Yt.T, mode='reduced')

        return Q

class GPUFastRandomizedRangeFinder(AbstractRangeFinder):
    def __init__(self, k=0, p=20):
        super().__init__(k, p, 0)

    def compute(self, A):
        m, n = A.shape

        # if self.k is set leave it as is, else set to min(m, n)
        self.k = self.k if self.k else min(m, n)

        R = random.sample(list(range(n)), self.k+self.p)
        D = cp.exp(2 * 1j * cp.random.sample(n) * np.pi)

        # Y.T is lxm => Y is mxl
        Yt = cp.sqrt(n/(self.k + self.p)) * cp.fft.fft(cp.multiply(D[:, None], A.T), norm='ortho', axis=0)[R, :]

        # Q is mxl ON matrix
        Q, _ = cp.linalg.qr(Yt.T, mode='reduced')

        return Q


class BlockRandomizedRangeFinder(AbstractRangeFinder):
    def __init__(self, k, p, q):
        super().__init__(k, p, q)

    def compute(self, A, col_num=2):
        m, n = A.shape

        # if self.k is set leave it as is, else set to min(m, n)
        self.k = self.k if self.k else min(m, n)

        # form n x (k+p) Gaussian random matrix G
        G = np.random.normal(size=(n, self.k+self.p))

        # number of columns we are pushing in the pipeline
        current_column = 0

        Y = np.zeros((m, self.k+self.p))

        while current_column < n:
            a = A[:, current_column:current_column + col_num]
            g = G[current_column:current_column + col_num]

            y = a @ g

            if self.q == 0:
                Y += y

            for _ in range(self.q):
                Y += np.outer(a, a) @ y

            current_column += col_num

        # Here we have all the columns and Y should be A @ G
        assert np.allclose(Y, A @ G)

        Q, _ = np.linalg.qr(Y, mode='reduced')

        return Q


class GPUBlockRandomizedRangeFinder(AbstractRangeFinder):
    def __init__(self, k, p, q):
        super().__init__(k, p)

    def compute(self, A, col_num=10):
        import cupy as cp
        m, n = A.shape

        # if self.k is set leave it as is, else set to min(m, n)
        self.k = self.k if self.k else min(m, n)

        # form n x (k+p) Gaussian random matrix G
        G = np.random.normal(size=(n, self.k+self.p))

        # number of columns we are pushing in the pipeline
        current_column = 0

        Y = cp.zeros((m, self.k+self.p))

        for current_column in range(0, n, col_num):
            a = cp.array(A[:, current_column:current_column + col_num])
            g = cp.array(G[current_column:current_column + col_num])

            y = a @ g

            if self.q == 0:
                Y += y

            for _ in range(self.q):
                Y += cp.outer(a, a) @ y

        # Here we have all the columns and Y should be A @ G
        # assert cp.allclose(Y, A @ G)

        Q, _ = cp.linalg.qr(Y, mode='reduced')

        return Q
