import unittest
import numpy as np

from randomized_algorithms import (BlockRandomizedRangeFinder, RandomizedRangeFinder,
                                   RandomizedSubspaceIteration, FastRandomizedRangeFinder)


class TestRangeFinder(unittest.TestCase):
    def check_error_bound(self, A, Q):
        m, n = A.shape
        _, k = Q.shape

        Omega = np.random.normal(size=(n, k))
        U, D, Vh = np.linalg.svd(A, full_matrices=False)

        Omega_1 = Vh[:k] @ Omega
        Omega_2 = Vh[k:] @ Omega

        Sigma_1 = np.diagflat(D[:k])
        Sigma_2 = np.diagflat(D[k:])

        return np.linalg.norm(A - Q @ Q.conj().T @ A)**2 <= (
                (np.linalg.norm(Sigma_2)**2) +
                (np.linalg.norm(Sigma_2 @ Omega_2 @ np.linalg.pinv(Omega_1)))**2)



m = 2048
n = 1024
k = n-10

class TestRandomizedRangeFinder(TestRangeFinder, unittest.TestCase):
    def test_error_bound(self):
        A = np.random.normal(size=(m, n))

        Q = RandomizedRangeFinder(A, k=k, p=0, q=0)

        self.assertTrue(self.check_error_bound(A, Q))


class TestRandomizedSubspaceIteration(TestRangeFinder, unittest.TestCase):
    def test_error_bound(self):
        A = np.random.normal(size=(m, n))

        Q = RandomizedSubspaceIteration(A, k=k, p=0, q=0)

        self.assertTrue(self.check_error_bound(A, Q))


class TestFastRandomizedRangeFinder(unittest.TestCase):
    def test_error_bound(self):
        A = np.random.normal(size=(m, n))

        Y = FastRandomizedRangeFinder(A, k=k, p=0)


class TestBlockRandomizedRangeFinder(TestRangeFinder, unittest.TestCase):
    def test_error_bound(self):
        A = np.random.normal(size=(m, n))

        Q = BlockRandomizedRangeFinder(A, k=k, p=0, q=0)

        self.assertTrue(self.check_error_bound(A, Q))

if __name__ == "__main__":
    unittest.main()
