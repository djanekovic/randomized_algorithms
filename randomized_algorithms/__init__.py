
# import CPU functions
from .range_finder import (BlockRandomizedRangeFinder, RandomizedRangeFinder,
                          RandomizedSubspaceIteration, FastRandomizedRangeFinder)
from .svd import (DirectSVD, RowExtraction, DirectEigenvalueDecomposition,
                  NystromMethod, SinglePassEigenvalueDecomposition)

# import GPU functions
from .range_finder import (GPURandomizedRangeFinder, GPUFastRandomizedRangeFinder)

from .svd import GPUDirectSVD

# import KSVD function

from .ksvd import RandomizedKSVD, ApproximateKSVD, ClassicKSVD


class BaseRSVD:
    def __init__(self, A, k, range_finder, factorization, q=0, eigh=False, debug=True):
        rf = range_finder(A, k, q=q, debug=debug)
        f = factorization(rf.Q, A=A, G=rf.G, Y=rf.Y, eigh=eigh, debug=debug)
        self._duration = dict(rf.duration, **f.duration)
        self._U = f.U
        self._D = f.D
        self._Vh = f.Vh

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
