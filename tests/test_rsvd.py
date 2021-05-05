import unittest
import numpy as np

from randomized_algorithms import *

m = 2048
n = 1024
k = 512

class TestSVDComputation():
    def check_svd_computation(self, A, U_, D_, Vh_):
        U, D, Vh = np.linalg.svd(A, full_matrices=False)
        A_ = U_ @ np.diagflat(D_) @ Vh_

        print (D[k+1], np.linalg.norm(A - A_, 2))

        return True

class DirectSVDTestCase(TestSVDComputation, unittest.TestCase):
    def test_randomized_range_finder(self):
        A = np.random.randn(m, n)
        Q = RandomizedRangeFinder(A, k)

        U, D, Vh = DirectSVD(A, Q)

        self.assertTrue(self.check_svd_computation(A, U, D, Vh))

    def test_randomized_subspace_iteration(self):
        A = np.random.randn(m, n)
        Q = RandomizedSubspaceIteration(A, k)

        U, D, Vh = DirectSVD(A, Q)

        self.assertTrue(self.check_svd_computation(A, U, D, Vh))

    def test_fast_randomized_range_finder(self):
        A = np.random.randn(m, n)
        Q, _ = np.linalg.qr(FastRandomizedRangeFinder(A, k))

        U, D, Vh = DirectSVD(A, Q)

        self.assertTrue(self.check_svd_computation(A, U, D, Vh))


    def test_block_randomized_range_finder(self):
        A = np.random.randn(m, n)
        Q = BlockRandomizedRangeFinder(A, k, col_num=512)

        U, D, Vh = DirectSVD(A, Q)

        self.assertTrue(self.check_svd_computation(A, U, D, Vh))

class DirectEigenvalueDecompositionTestCase(TestSVDComputation, unittest.TestCase):
    def test_randomized_range_finder(self):
        A = np.random.randn(m, n)
        A = A @ A.conj().T

        Q = RandomizedRangeFinder(A, k)

        U, D, Vh = DirectEigenvalueDecomposition(A, Q)

        self.assertTrue(self.check_svd_computation(A, U, D, Vh))

    def test_randomized_subspace_iteration(self):
        A = np.random.randn(m, n)
        A = A @ A.conj().T

        Q = RandomizedSubspaceIteration(A, k)

        U, D, Vh = DirectEigenvalueDecomposition(A, Q)

        self.assertTrue(self.check_svd_computation(A, U, D, Vh))

    def test_fast_randomized_range_finder(self):
        A = np.random.randn(m, n)
        A = A @ A.conj().T

        Q, _ = np.linalg.qr(FastRandomizedRangeFinder(A, k))

        U, D, Vh = DirectEigenvalueDecomposition(A, Q)

        self.assertTrue(self.check_svd_computation(A, U, D, Vh))


    def test_block_randomized_range_finder(self):
        A = np.random.randn(m, n)
        A = A @ A.conj().T

        Q = BlockRandomizedRangeFinder(A, k, col_num=512)

        U, D, Vh = DirectEigenvalueDecomposition(A, Q)

        self.assertTrue(self.check_svd_computation(A, U, D, Vh))


class SinglePassEigenvalueDecompositionTestCase(TestSVDComputation, unittest.TestCase):
    def test_randomized_range_finder_qr(self):
        A = np.random.randn(m, n)
        A = A @ A.conj().T

        Q, G, Y = RandomizedRangeFinder(A, k, return_random_and_sample=True)

        V, D = SinglePassEigenvalueDecomposition(G, Q, Y)

        self.assertTrue(self.check_svd_computation(A, V, D, V.conj().T))

    def test_randomized_subspace_iteration(self):
        A = np.random.randn(m, n)
        A = A @ A.conj().T

        Q, G, Y = RandomizedSubspaceIteration(A, k, return_random_and_sample=True)

        V, D = SinglePassEigenvalueDecomposition(G, Q, Y)

        self.assertTrue(self.check_svd_computation(A, V, D, V.conj().T))

    def test_block_randomized_range_finder(self):
        A = np.random.randn(m, n)
        A = A @ A.conj().T

        Q, G, Y = BlockRandomizedRangeFinder(A, k, col_num=512,
                                             return_random_and_sample=True)

        V, D = SinglePassEigenvalueDecomposition(G, Q, Y)

        self.assertTrue(self.check_svd_computation(A, V, D, V.conj().T))

if __name__ == "__main__":
    unittest.main()
