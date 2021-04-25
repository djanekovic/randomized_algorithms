import rsvd
import utils
import time

import numpy as np

def timeit(num_times=1):
    def decorator_timeit(func):
        def wrapper_timeit(*args, **kwargs):
            start = time.time()
            for _ in range(num_times):
                func(*args, *kwargs)
            duration = time.time() - start
            return duration/num_times
        return wrapper_timeit
    return decorator_timeit


@timeit(num_times=100)
def single_pass_rsvd_benchmark(A, k, p):
    u, d = rsvd.single_pass_rsvd(A, k=k, p=p)

def rsvd_benchmark(A, k, p):
    u, d, vh = rsvd.rsvd_basic(A, k=k, p=p)

@timeit(num_times=100)
def numpy_svd_benchmark(A):
    u, d, vh = np.linalg.svd(A, full_matrices=False)

if __name__ == "__main__":
    A = utils.generate_random_symmetric_matrix(500, 200)
    print("RSVD {}: {}".format(A.shape, single_pass_rsvd_benchmark(A, 190, 10)))
    print("Numpy SVD {}: {}".format(A.shape, numpy_svd_benchmark(A)))

    u, d = rsvd.single_pass_rsvd(A, 190, 10)

    print(np.linalg.norm(A - (u @ np.diag(d)) @ u.H))
