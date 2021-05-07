import numpy as np
import cupy as cp
from cupyx.time import repeat
import time
import csv

from randomized_algorithms import *

def direct_svd(A_cpu, k, q, check_finite=False, debug=False, eigh=False):
    # Transfer A to the GPU
    A_gpu = cp.asarray(A_cpu)

    # Compute Q_gpu with A_gpu

    Q_gpu = GPURandomizedRangeFinder(A_gpu, k=k, q=q, debug=debug)

    # Compute small SVD with A_gpu and Q_gpu
    U_gpu, D_gpu, Vh_gpu = GPUDirectSVD(A_gpu, Q_gpu, eigh=eigh)

    # move result from GPU to the CPU
    return U_gpu.get(), D_gpu.get(), Vh_gpu.get()


def fast_svd(A_cpu, k, debug=False):
    # Transfer A to the GPU
    A_gpu = cp.asarray(A_cpu)

    # Compute Q_gpu with A_gpu
    Q_gpu, _ = cp.linalg.qr(GPUFastRandomizedRangeFinder(A_gpu, k=k, debug=debug))

    # Compute small SVD with A_gpu and Q_gpu
    U_gpu, D_gpu, Vh_gpu = GPUDirectSVD(A_gpu, Q_gpu)

    # move result from GPU to the CPU
    return U_gpu.get(), D_gpu.get(), Vh_gpu.get()


def baseline_gpu_svd(A_cpu):
    # Transfer A to the GPU
    A_gpu = cp.asarray(A_cpu)

    # Compute SVD on the GPU
    U_gpu, D_gpu, Vh_gpu = cp.linalg.svd(A_gpu, full_matrices=False)

    # Return CPU matrices
    return U_gpu.get(), D_gpu.get(), Vh_gpu.get()


def baseline_cpu_svd(A_cpu):
    return np.linalg.svd(A_cpu, full_matrices=False)


def time_wrapper(func, num_times=1, args=(), kwargs={}):
    start = time.time()

    [ func(*args, **kwargs) for i in range(num_times)]

    return (time.time() - start) / num_times


def approx_norm(A, U, D, Vh):
    return np.linalg.norm(A -  np.dot(U * D,  Vh), 2)

def generate_speed_data(m, n, num_times, ranks):
    A = np.random.normal(size=(m, n))
    cpu_svd_time = time_wrapper(baseline_cpu_svd, num_times=1,
                                args=(A,))
    gpu_svd_time = repeat(baseline_gpu_svd, n_repeat=num_times,
                          args=(A,))

    print (gpu_svd_time)

    direct_svd_times = []
    direct_svd_eigh_times = []
    fast_svd_times = []

    # for each rank
    for q in range(2):
        direct_svd_times_q = []
        direct_svd_eigh_times_q = []

        for k in ranks:
            direct_svd_times_q.append(repeat(direct_svd, args=(A, k, q), n_repeat=num_times))
            direct_svd_eigh_times_q.append(repeat(direct_svd, args=(A, k, q),
                                                  kwargs={'eigh': True}, n_repeat=num_times))

        direct_svd_times.append(direct_svd_times_q)
        direct_svd_eigh_times.append(direct_svd_eigh_times_q)

    for k in ranks:
        fast_svd_times.append(repeat(fast_svd, args=(A, k), n_repeat=num_times))

    return cpu_svd_time, gpu_svd_time, direct_svd_times, direct_svd_eigh_times, fast_svd_times

def write_out_speed_data(m, n, num_times, ranks, factor):
    print("Generating speed data for factor {}...".format(factor))
    cpu_svd, gpu_svd, direct_svd, svd_eigh, fast_svd = generate_speed_data(m, n, num_times, ranks)

    print("Writing out CSV file for speed data...")

    with open("speed_data_{}.csv".format(factor), 'w', newline='') as f:
        speed_writer = csv.writer(f)
        speed_writer.writerow(["Ranks", "DirectSVD q=0", "DirectSVD q=1",
                                        "EighSVD q=0", "EighSVD q=1",
                                        "FastSVD", "CPU Baseline", "GPU Baseline"])
        for (rank, *gpu_times) in zip(ranks, direct_svd[0], direct_svd[1], svd_eigh[0], svd_eigh[1], fast_svd):
            speed_writer.writerow([rank] + [np.average(i.gpu_times) for i in gpu_times] +
                                  [np.average(gpu_svd.gpu_times), cpu_svd])


if __name__ == "__main__":
    cp.cuda.Device(1).use()
    for factor in range(1, 3):
        (m, n) = (1024 * factor, 512 * factor)
        num_times = 2
        # unique is extra here but just to be sure
        ranks = np.unique(np.linspace(1, n, 15, endpoint=False, dtype=np.int32))

        write_out_speed_data(m, n, num_times, ranks, factor)
