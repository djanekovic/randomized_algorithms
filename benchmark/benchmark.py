from randomized_algorithms import *

from itertools import product
from time import time

import numpy as np
import os
import csv

KEYS = ['GEMM', 'QR', 'Factorization']

class BenchmarkRun:
    def __init__(self, A, k, range_finder, factorization,
                 q=0, eigh=False, num_times=1, debug=False):
        dicts = [
            BaseRSVD(A, k, range_finder, factorization, q=q, eigh=eigh, debug=debug).duration
                for _ in range(num_times)
        ]

        # aggregate duration dicts
        self._duration_dicts = { k: sum([d[k] for d in dicts])/num_times for k in KEYS }

        self._total_duration_avg = sum([sum(d.values()) for d in dicts])

    @property
    def duration_dicts(self):
        return self._duration_dicts

    @property
    def total_duration_avg(self):
        return self._total_duration_avg


def DirctSVDBenchmarkGenerator(A, k, q=0, eigh=False, num_times=1, debug=False):
    return BenchmarkRun(A, k, RandomizedRangeFinder, DirectSVD,
                        q=q, eigh=eigh, num_times=num_times, debug=debug)


def FastDirectSVDBenchmarkGenerator(A, k, num_times=1, debug=False):
    return BenchmarkRun(A, k, FastRandomizedRangeFinder, DirectSVD,
                        num_times=num_times, debug=debug)


def direct_svd(A, k, q=0, eigh=False):
    rsvd = BaseRSVD(A, k, RandomizedRangeFinder, DirectSVD, q, eigh)
    return rsvd.U, rsvd.D, rsvd.Vh


def fast_svd(A, k, debug=False):
    rsvd = BaseRSVD(A, k, FastRandomizedRangeFinder, DirectSVD, debug=debug)
    return rsvd.U, rsvd.D, rsvd.Vh


def time_wrapper(func, num_times, *args, **kwargs):
    start = time()

    [ func(*args, **kwargs) for i in range(num_times)]

    return (time() - start) / num_times


def approx_norm(A, U, D, Vh):
    return np.linalg.norm(A -  np.dot(U * D,  Vh), 2)


def generate_speed_data(m, n, num_times, ranks, q_range=2):
    A = np.random.normal(size=(m, n))
    baseline_svd_time = time_wrapper(np.linalg.svd, num_times, A, full_matrices=False)

    # for each rank
    direct_svd_times = [
        [DirctSVDBenchmarkGenerator(A, k, q=q, eigh=False, num_times=num_times).duration_dicts
            for k in ranks] for q in range(q_range)
    ]


    direct_svd_eigh_times = [
        [DirctSVDBenchmarkGenerator(A, k, q=q, eigh=True, num_times=num_times).duration_dicts
            for k in ranks] for q in range(q_range)
    ]

    fast_svd_times = [
        FastDirectSVDBenchmarkGenerator(A, k, num_times).duration_dicts for k in ranks
    ]

    return baseline_svd_time, direct_svd_times, direct_svd_eigh_times, fast_svd_times


def write_out_speed_dicts(f, durations):
    speed_writer = csv.writer(f)
    speed_writer.writerow(["Ranks", *KEYS])
    [
        speed_writer.writerow([k, *map(d.get, KEYS)]) for k, d in zip(ranks, durations)
    ]

def write_out_speed_data(m, n, num_times, ranks):
    print("Generating speed data...")
    baseline, direct_svd, direct_svd_eigh, fast_svd = generate_speed_data(m, n, num_times, ranks)

    print("Writing out CSV file for speed data...")

    os.makedirs('speed_data', exist_ok=True)

    for q, durations in enumerate(direct_svd):
        with open('speed_data/DirectSVD_q_{}.csv'.format(q), 'w', newline='') as f:
            write_out_speed_dicts(f, durations)

    for q, durations in enumerate(direct_svd_eigh):
        with open('speed_data/DirectEighSVD_q_{}.csv'.format(q), 'w', newline='') as f:
            write_out_speed_dicts(f, durations)

    with open('speed_data/FastSVD.csv'.format(q), 'w', newline='') as f:
        write_out_speed_dicts(f, fast_svd)

    with open('speed_data/Baseline.txt', 'w', newline='') as f:
        f.write(str(baseline))


def generate_accuracy_data(m, n, num_times, logspace_stop, ranks, q_range=2):
    A = np.random.normal(size=(m, n))

    U, _, Vh = np.linalg.svd(A, full_matrices=False)
    D = np.logspace(0, logspace_stop, min(m, n))
    A = np.dot(U * D, Vh)

    direct_svd_norms = [
        [approx_norm(A, *direct_svd(A, k=k, q=q, eigh=False)) for k in ranks]
            for q in range(q_range)
    ]

    direct_svd_eigh_norms = [
        [approx_norm(A, *direct_svd(A, k=k, q=q, eigh=True)) for k in ranks]
            for q in range(q_range)
    ]

    fast_svd_norms = [approx_norm(A, *fast_svd(A, k=k)) for k in ranks]

    return D[ranks], direct_svd_norms, direct_svd_eigh_norms, fast_svd_norms


def write_out_accuracy_data(m, n, num_times, ranks):
    os.makedirs('accuracy_data', exist_ok=True)

    for logspace_stop in (-0.5, -1, -2, -3.5):
        print("Generating accuracy data for logspace_stop: {}...".format(logspace_stop))
        sing_vals, direct_svd, direct_svd_eigh, fast_svd = generate_accuracy_data(m, n, num_times, logspace_stop, ranks)

        print ("Writing out CSV file for accuracy data...")

        with open("accuracy_data/logspace_{}.csv".format(logspace_stop), 'w', newline='') as f:
            acc_writer = csv.writer(f)
            acc_writer.writerow(["Ranks", "DirectSVD q=0", "DirectSVD q=1",
                                               "EighSVD q=0", "EighSVD q=1",
                                               "FastSVD", "Baseline"])
            [acc_writer.writerow(row) for row in zip(ranks, *[d for d in direct_svd],
                                                     *[d for d in direct_svd_eigh],
                                                     fast_svd, sing_vals)]


if __name__ == "__main__":
    (m, n) = (512 * 2, 256 * 2)
    num_times = 1
    # unique is extra here but just to be sure
    ranks = np.unique(np.linspace(1, n, 10, endpoint=False, dtype=np.int32))

    write_out_speed_data(m, n, num_times, ranks)
    write_out_accuracy_data(m, n, num_times, ranks)
