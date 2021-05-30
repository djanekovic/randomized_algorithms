import unittest
import numpy as np

from randomized_algorithms import *

m = 2048
n = 1024
k = 512

class TestSVDComputation(unittest.TestCase):
    def tearDown(self):
        D = np.linalg.svd(self.A, full_matrices=False, compute_uv=False)
        A_ = np.dot(self._rsvd.U *self._rsvd.D, self._rsvd.Vh)
        print (D[k+1], np.linalg.norm(self.A - A_, 2))

    def directSVDGenerator(self, range_finder):
        return BaseRSVD(self.A, k, range_finder, DirectSVD)

    def directEigenvalueDecompositionGenerator(self, range_finder):
        return BaseRSVD(self.A, k, range_finder, DirectEigenvalueDecomposition)

    def singlePassEigenvalueDecompositionGenerator(self, range_finder):
        return BaseRSVD(self.A, k, range_finder, SinglePassEigenvalueDecomposition)


class TestGeneralSVDComputation(TestSVDComputation):
    def setUp(self):
        self.A = np.random.randn(m, n)

class TestHermiteanSVDComputation(TestSVDComputation):
    def setUp(self):
        A = np.random.randn(m, n)
        self.A = A @ A.conj().T


class DirectSVDTestCase(TestGeneralSVDComputation):
    def test_randomized_range_finder(self):
        self._rsvd = self.directSVDGenerator(RandomizedRangeFinder)


    def test_randomized_subspace_iteration(self):
        self._rsvd = self.directSVDGenerator(RandomizedSubspaceIteration)


    def test_fast_randomized_range_finder(self):
        self._rsvd = self.directSVDGenerator(FastRandomizedRangeFinder)


    def test_block_randomized_range_finder(self):
        self._rsvd = self.directSVDGenerator(BlockRandomizedRangeFinder)

class DirectEigenvalueDecompositionTestCase(TestHermiteanSVDComputation):
    def test_randomized_range_finder(self):
        self._rsvd = self.directEigenvalueDecompositionGenerator(RandomizedRangeFinder)


    def test_randomized_subspace_iteration(self):
        self._rsvd = self.directEigenvalueDecompositionGenerator(RandomizedSubspaceIteration)


    def test_fast_randomized_range_finder(self):
        self._rsvd = self.directEigenvalueDecompositionGenerator(FastRandomizedRangeFinder)


    def test_block_randomized_range_finder(self):
        self._rsvd = self.directEigenvalueDecompositionGenerator(BlockRandomizedRangeFinder)


class SinglePassEigenvalueDecompositionTestCase(TestHermiteanSVDComputation):
    def test_randomized_range_finder_qr(self):
        self._rsvd = self.singlePassEigenvalueDecompositionGenerator(RandomizedRangeFinder)

    def test_randomized_subspace_iteration(self):
        self._rsvd = self.singlePassEigenvalueDecompositionGenerator(RandomizedSubspaceIteration)

    def test_fast_randomized_range_finder(self):
        self._rsvd = self.singlePassEigenvalueDecompositionGenerator(FastRandomizedRangeFinder)

    def test_block_randomized_range_finder(self):
        self._rsvd = self.singlePassEigenvalueDecompositionGenerator(BlockRandomizedRangeFinder)

if __name__ == "__main__":
    unittest.main()
