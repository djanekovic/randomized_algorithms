import unittest
import numpy as np
from functools import wraps

from randomized_algorithms import (BlockRandomizedRangeFinder, RandomizedRangeFinder,
                                   RandomizedSubspaceIteration, FastRandomizedRangeFinder)


m = 2048
n = 1024
k = n-10


class BaseRandomizedRangeFinder(unittest.TestCase):
    def setUp(self):
        self.A = np.random.randn(m, n)

    def tearDown(self):
        Omega = self._rf.G
        Q = self._rf.Q

        U, D, Vh = np.linalg.svd(self.A, full_matrices=False)

        Omega_1 = Vh[:k] @ Omega
        Omega_2 = Vh[k:] @ Omega

        Sigma_1 = np.diagflat(D[:k])
        Sigma_2 = np.diagflat(D[k:])

        condition = np.linalg.norm(self.A - Q @ Q.conj().T @ self.A)**2 <= (
                (np.linalg.norm(Sigma_2)**2) +
                (np.linalg.norm(Sigma_2 @ Omega_2 @ np.linalg.pinv(Omega_1)))**2)

        self.assertTrue(condition)


class TestRandomizedRangeFinder(BaseRandomizedRangeFinder):
    def test_error_bound(self):
        self._rf = RandomizedRangeFinder(self.A, k=k, p=0, q=0)


class TestRandomizedSubspaceIteration(BaseRandomizedRangeFinder):
    def test_error_bound(self):
        self._rf = RandomizedSubspaceIteration(self.A, k=k, p=0, q=0)


class TestBlockRandomizedRangeFinder(BaseRandomizedRangeFinder):
    def test_error_bound(self):
        self._rf = BlockRandomizedRangeFinder(self.A, k=k, p=0, q=0)

class TestFastRandomizedRangeFinder(BaseRandomizedRangeFinder):
    def test_error_bound(self):
        self._rf = FastRandomizedRangeFinder(self.A, k=k, p=0)

if __name__ == "__main__":
    unittest.main()
