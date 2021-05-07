from randomized_algorithms import *
from functools import partial
import numpy as np
import time
import csv
from itertools import cycle


def direct_eigh(A, k, q=0, check_finite=False, debug=False, eigh=False):
    Q = RandomizedRangeFinder(A, k=k, q=q, check_finite=check_finite, debug=debug)
    return DirectEigenvalueDecomposition(A, Q, debug=debug)


def single_pass(A, k, q=0, check_finite=False, debug=False):
    Q, G, Y = RandomizedRangeFinder(A, k=k, q=q, check_finite=check_finite, debug=debug,
                                    return_random_and_sample=True)
    V, D = SinglePassEigenvalueDecomposition(Q, G, Y, debug=debug)

    return V, D, V.conj().T


def time_wrapper(func, num_times):
    start = time.time()

    [func() for i in range(num_times)]

    return (time.time() - start) / num_times


def approx_norm(A, U, D, Vh):
    return np.linalg.norm(A -  np.dot(U * D,  Vh), 2)


def generate_speed_data(m, n, num_times, ranks):
    A = np.random.normal(size=(m, n))
    A = A @ A.T

    baseline_svd_partial = partial(np.linalg.svd, A, hermitian=True, full_matrices=False)
    baseline_svd_time = time_wrapper(baseline_svd_partial, num_times)

    direct_eigh_times = []
    single_pass_eigh_times = []

    # for each rank
    for q in range(2):
        direct_eigh_times_q = []

        for k in ranks:
            direct_eigh_partial = partial(direct_eigh, A, k=k, q=q)

            direct_eigh_times_q.append(time_wrapper(direct_eigh_partial, num_times))

        direct_eigh_times.append(direct_eigh_times_q)

    for k in ranks:
        single_pass_eigh_partial = partial(single_pass, A, k=k)
        single_pass_eigh_times.append(time_wrapper(single_pass_eigh_partial, num_times))

    return baseline_svd_time, direct_eigh_times, single_pass_eigh_times


def generate_accuracy_data(m, n, num_times, logspace_stop, ranks):
    A = np.random.normal(size=(m, n))
    U, D, Vh = np.linalg.svd(A, full_matrices=False)
    D = D[0] * np.logspace(0, logspace_stop, n)
    A = np.dot(U * D, Vh)

    A = A @ A.T

    D = np.linalg.svd(A, compute_uv=False, full_matrices=False, hermitian=True)

    direct_eigh_norms = []
    single_pass_eigh_norms = []

    for q in range(2):
        direct_eigh_norms_q = [approx_norm(A, *direct_eigh(A, k=k, q=q, debug=True)) for k in ranks]
        direct_eigh_norms.append(direct_eigh_norms_q)

    single_pass_eigh_norms = [approx_norm(A, *single_pass(A, k=k, debug=True)) for k in ranks]

    return D[ranks], direct_eigh_norms, single_pass_eigh_norms



def write_out_speed_data(m, n, num_times, ranks):
    print("Generating speed data...")
    baseline, direct_eigh, single_pass_eigh = generate_speed_data(m, n, num_times, ranks)

    print("Writing out CSV file for speed data...")

    with open("speed_data.csv", 'w', newline='') as f:
        speed_writer = csv.writer(f)
        speed_writer.writerow(["Ranks", "DirectEigh q=0", "DirectEigh q=1",
                                 "SinglePassEigh", "Baseline"])
        [speed_writer.writerow([*row, baseline]) for row in zip(ranks,
                                                                direct_eigh[0], direct_eigh[1],
                                                                single_pass_eigh)]


def write_out_accuracy_data(m, n, num_times, ranks):
    for logspace_stop in (-0.5, -1, -2, -3.5):
        print("Generating accuracy data for logspace_stop: {}...".format(logspace_stop))
        sing_vals, direct_eigh, single_pass_eigh = generate_accuracy_data(m, n, num_times, logspace_stop, ranks)

        print ("Writing out CSV file for accuracy data...")

        with open("accuracy_data_{}.csv".format(logspace_stop), 'w', newline='') as f:
            acc_writer = csv.writer(f)
            acc_writer.writerow(["Ranks", "DirectEigh q=0", "DirectEigh q=1",
                                 "SinglePassEigh", "Baseline"])
            [acc_writer.writerow(row) for row in zip(ranks,
                                                     direct_eigh[0], direct_eigh[1],
                                                     single_pass_eigh, sing_vals)]


if __name__ == "__main__":
    (m, n) = (1024 * 1, 1024 * 1)
    num_times = 1
    # unique is extra here but just to be sure
    ranks = np.unique(np.linspace(1, n, 10, endpoint=False, dtype=np.int32))

    write_out_speed_data(m, n, num_times, ranks)
    write_out_accuracy_data(m, n, num_times, ranks)
