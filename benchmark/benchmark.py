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


def DirectSVDBenchmarkGenerator(A, k, q=0, eigh=False, num_times=1, debug=False):
    return BenchmarkRun(A, k, RandomizedRangeFinder, DirectSVD,
                        q=q, eigh=eigh, num_times=num_times, debug=debug)


def FastDirectSVDBenchmarkGenerator(A, k, num_times=1, debug=False):
    return BenchmarkRun(A, k, FastRandomizedRangeFinder, DirectSVD,
                        num_times=num_times, debug=debug)

def DirectEighBenchmarkGenerator(A, k, q=0, num_times=1, debug=False):
    return BenchmarkRun(A, k, RandomizedRangeFinder, DirectEigh,
                        num_times=num_times, debug=debug)

def SinglePassEighBenchmarkGenerator(A, k, num_times=1, debug=False):
    return BenchmarkRun(A, k, RandomizedRangeFinder, SinglePassEigh,
                        num_times=num_times, debug=debug)

def FastDirectEighBenchmarkGenerator(A, k, num_times=1, debug=False):
    return BenchmarkRun(A, k, FastRandomizedRangeFinder, DirectEigh,
                        num_times=num_times, debug=debug)


def direct_eigh(A, k, q):
    rsvd = BaseRSVD(A, k, RandomizedRangeFinder, DirectSVD, q)
    return rsvd.U, rsvd.D, rsvd.Vh

def single_pass_eigh(A, k):
    rsvd = BaseRSVD(A, k, RandomizedRangeFinder, SinglePassEigh)
    return rsvd.U, rsvd.D, rsvd.Vh

def fast_eigh(A, k):
    rsvd = BaseRSVD(A, k, FastRandomizedRangeFinder, DirectEigh)
    return rsvd.U, rsvd.D, rsvd.Vh


def direct_svd(A, k, q=0, eigh=False):
    rsvd = BaseRSVD(A, k, RandomizedRangeFinder, DirectSVD, q, eigh)
    return rsvd.U, rsvd.D, rsvd.Vh


def fast_svd(A, k):
    rsvd = BaseRSVD(A, k, FastRandomizedRangeFinder, DirectSVD)
    return rsvd.U, rsvd.D, rsvd.Vh


def time_wrapper(func, num_times, *args, **kwargs):
    start = time()

    [ func(*args, **kwargs) for i in range(num_times)]

    return (time() - start) / num_times


def approx_norm(A, U, D, Vh):
    return np.linalg.norm(A -  np.dot(U * D,  Vh), 2)


class GeneralBenchmark():
    def __init__(self, m, n, num_times, ranks, q_range=2):
        self._m = m
        self._n = n
        self._num_times = num_times
        self._ranks = ranks
        self._q_range = q_range

        self.write_out_speed_data()
        self.write_out_accuracy_data()

    def _speed_base(self):
        return "speed_data/general/real/"

    def _accuracy_base(self):
        return "accuracy_data/general/real/"

    def _generate_matrix(self):
        return np.random.normal(size=(self._m, self._n))


    def _generate_speed_data(self):
        A = self._generate_matrix()
        baseline_svd_time = time_wrapper(np.linalg.svd, num_times, A, full_matrices=False)

        # for each rank
        direct_svd_times = [
            [DirectSVDBenchmarkGenerator(A, k, q=q, eigh=False, num_times=self._num_times).duration_dicts
                for k in self._ranks] for q in range(self._q_range)
        ]


        direct_svd_eigh_times = [
            [DirectSVDBenchmarkGenerator(A, k, q=q, eigh=True, num_times=self._num_times).duration_dicts
                for k in self._ranks] for q in range(self._q_range)
        ]

        fast_svd_times = [
            FastDirectSVDBenchmarkGenerator(A, k, self._num_times).duration_dicts for k in self._ranks
        ]

        return baseline_svd_time, direct_svd_times, direct_svd_eigh_times, fast_svd_times


    def _write_out_speed_dicts(self, f, durations):
        speed_writer = csv.writer(f)
        speed_writer.writerow(["Ranks", *KEYS])
        [speed_writer.writerow([k, *map(d.get, KEYS)]) for k, d in zip(self._ranks, durations)]

    def write_out_speed_data(self):
        print("Generating speed data...")
        baseline, direct_svd, direct_svd_eigh, fast_svd = self._generate_speed_data()

        print("Writing out CSV file for speed data...")

        base = self._speed_base()
        os.makedirs(base, exist_ok=True)
        for q, durations in enumerate(direct_svd):
            with open(base + 'DirectSVD_q_{}.csv'.format(q), 'w', newline='') as f:
                self._write_out_speed_dicts(f, durations)

        for q, durations in enumerate(direct_svd_eigh):
            with open(base + 'DirectEighSVD_q_{}.csv'.format(q), 'w', newline='') as f:
                self._write_out_speed_dicts(f, durations)

        with open(base + 'FastSVD.csv'.format(q), 'w', newline='') as f:
            self._write_out_speed_dicts(f, fast_svd)

        with open(base + 'Baseline.txt', 'w', newline='') as f:
            f.write(str(baseline))



    def _generate_accuracy_data(self, logspace_stop):
        A = self._generate_matrix()

        U, _, Vh = np.linalg.svd(A, full_matrices=False)
        D = np.logspace(0, logspace_stop, min(self._m, self._n))
        A = np.dot(U * D, Vh)

        direct_svd_norms = [
            [approx_norm(A, *direct_svd(A, k=k, q=q, eigh=False)) for k in self._ranks]
                for q in range(self._q_range)
        ]

        direct_svd_eigh_norms = [
            [approx_norm(A, *direct_svd(A, k=k, q=q, eigh=True)) for k in self._ranks]
                for q in range(self._q_range)
        ]

        fast_svd_norms = [approx_norm(A, *fast_svd(A, k=k)) for k in self._ranks]

        return D[ranks], direct_svd_norms, direct_svd_eigh_norms, fast_svd_norms



    def write_out_accuracy_data(self):
        base = self._accuracy_base()
        os.makedirs(base, exist_ok=True)
        for logspace_stop in (-0.5, -1, -2, -3.5):
            print("Generating accuracy data for logspace_stop: {}...".format(logspace_stop))
            sing_vals, direct_svd, direct_svd_eigh, fast_svd = self._generate_accuracy_data(logspace_stop)

            print ("Writing out CSV file for accuracy data...")
            with open(base + 'logspace_{}.csv'.format(logspace_stop), 'w', newline='') as f:
                acc_writer = csv.writer(f)
                acc_writer.writerow(["Ranks", "DirectSVD q=0", "DirectSVD q=1",
                                     "EighSVD q=0", "EighSVD q=1", "FastSVD", "Baseline"])
                [acc_writer.writerow(row) for row in zip(self._ranks, *[d for d in direct_svd],
                                                         *[d for d in direct_svd_eigh],
                                                         fast_svd, sing_vals)]



class GeneralComplexBenchmark(GeneralBenchmark):
    def __init__(self, m, n, num_times, ranks, q_range=2):
        super().__init__(m, n, num_times, ranks, q_range=q_range)

    def _speed_base(self):
        return "speed_data/general/complex/"

    def _accuracy_base(self):
        return "accuracy_data/general/complex/"

    def _generate_matrix(self):
        return np.random.normal(size=(self._m, self._n)) + 1j * np.random.normal(size=(self._m, self._n))


#TODO: yikes we are duplicating code, I know... Either don't inherit from GeneralBenchmark
# or redesign
class HermitianBenchmark(GeneralBenchmark):
    def __init__(self, m, n, num_times, ranks, q_range=2):
        super().__init__(m, n, num_times, ranks, q_range)

    def _speed_base(self):
        return "speed_data/hermitian/real/"

    def _accuracy_base(self):
        return "accuracy_data/hermitian/real/"

    def _generate_speed_data(self):
        A = self._generate_matrix()
        A = A @ A.conj().T

        baseline_svd_time = time_wrapper(np.linalg.eigh, num_times, A)

        # for each rank
        direct_eigh_times = [
            [DirectEighBenchmarkGenerator(A, k, q=q, num_times=self._num_times).duration_dicts
                for k in self._ranks] for q in range(self._q_range)
        ]


        single_pass_svd_times = [
            SinglePassEighBenchmarkGenerator(A, k, num_times=self._num_times).duration_dicts
                for k in self._ranks
        ]


        fast_svd_times = [
            FastDirectEighBenchmarkGenerator(A, k, self._num_times).duration_dicts
                for k in self._ranks
        ]

        return baseline_svd_time, direct_eigh_times, single_pass_svd_times, fast_svd_times


    def _write_out_speed_dicts(self, f, durations):
        speed_writer = csv.writer(f)
        speed_writer.writerow(["Ranks", *KEYS])
        [speed_writer.writerow([k, *map(d.get, KEYS)]) for k, d in zip(self._ranks, durations)]

    def write_out_speed_data(self):
        print("Generating speed data...")
        baseline, direct, single_pass, fast = self._generate_speed_data()

        print("Writing out CSV file for speed data...")

        base = self._speed_base()
        os.makedirs(base, exist_ok=True)
        for q, durations in enumerate(direct):
            with open(base + '/DirectEigh_q_{}.csv'.format(q), 'w', newline='') as f:
                self._write_out_speed_dicts(f, durations)

        with open(base + '/SinglePassEigh.csv'.format(q), 'w', newline='') as f:
            self._write_out_speed_dicts(f, single_pass)

        with open(base + '/FastSVD.csv'.format(q), 'w', newline='') as f:
            self._write_out_speed_dicts(f, fast)

        with open(base + '/Baseline.txt', 'w', newline='') as f:
            f.write(str(baseline))



    def _generate_accuracy_data(self, logspace_stop):
        A = self._generate_matrix()
        U, D, Vh = np.linalg.svd(A, full_matrices=False)
        D = D[0] * np.logspace(0, logspace_stop, self._n)
        A = np.dot(U * D, Vh)

        A = A @ A.conj().T

        D = np.linalg.svd(A, compute_uv=False, full_matrices=False, hermitian=True)

        direct_eigh_norms = [
            [approx_norm(A, *direct_eigh(A, k=k, q=q)) for k in self._ranks]
                for q in range(self._q_range)
        ]

        single_pass_norms = [approx_norm(A, *single_pass_eigh(A, k=k)) for k in self._ranks]
        fast_svd_norms = [approx_norm(A, *fast_eigh(A, k=k)) for k in self._ranks]

        return D[ranks], direct_eigh_norms, single_pass_norms, fast_svd_norms



    def write_out_accuracy_data(self):
        base = self._accuracy_base()
        os.makedirs(base, exist_ok=True)
        for logspace_stop in (-0.5, -1, -2, -3.5):
            print("Generating accuracy data for logspace_stop: {}...".format(logspace_stop))
            sing_vals, direct, single_pass, fast = self._generate_accuracy_data(logspace_stop)

            print ("Writing out CSV file for accuracy data...")

            with open(base + '/logspace_{}.csv'.format(logspace_stop), 'w', newline='') as f:
                acc_writer = csv.writer(f)
                acc_writer.writerow(["Ranks", "DirectEigh q=0", "DirectEigh q=1",
                                     "SinglePass", "FastSVD", "Baseline"])
                [acc_writer.writerow(row) for row in zip(self._ranks, *[d for d in direct],
                                                         single_pass, fast, sing_vals)]


class HermitianComplexBenchmark(HermitianBenchmark):
    def __init__(self, m, n, num_times, ranks, q_range=2):
        super().__init__(m, n, num_times, ranks, q_range=q_range)

    def _speed_base(self):
        return "speed_data/hermitian/complex/"

    def _accuracy_base(self):
        return "accuracy_data/hermitian/complex/"

    def _generate_matrix(self):
        return np.random.normal(size=(self._m, self._n)) + 1j * np.random.normal(size=(self._m, self._n))



if __name__ == "__main__":
    (m, n) = (512, 256)
    num_times = 1
    # unique is extra here but just to be sure
    ranks = np.unique(np.linspace(1, n, 10, endpoint=False, dtype=np.int32))

    benchmark = GeneralBenchmark(m, n, num_times, ranks)
    benchmark = GeneralComplexBenchmark(m, n, num_times, ranks)
    benchmark = HermitianBenchmark(m, n, num_times, ranks)
    benchmark = HermitianComplexBenchmark(m, n, num_times, ranks)
