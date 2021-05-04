from randomized_algorithms import *
from functools import partial
import numpy as np
import time
import csv
from itertools import cycle
from matplotlib import pyplot as plt

def direct_svd(A, k, q=0, check_finite=False, debug=False, eigh=False):
    Q = RandomizedRangeFinder(A, k=k, q=q, check_finite=check_finite, debug=debug)
    return DirectSVD(A, Q, debug=debug, check_finite=check_finite, eigh=eigh)


def fast_svd(A, k, debug=False):
    Q, _ = np.linalg.qr(FastRandomizedRangeFinder(A, k=k, debug=debug))
    return DirectSVD(A, Q, debug=debug, check_finite=False)


def baseline_svd(A):
    return np.linalg.svd(A, full_matrices=False)


def time_wrapper(func, num_times):
    start = time.time()

    [ func() for i in range(num_times)]

    return (time.time() - start) / num_times


def approx_norm(A, U, D, Vh):
    return np.linalg.norm(A -  np.dot(U * D,  Vh), 2)

def generate_speed_data(m, n, num_times, ranks):
    A = np.random.normal(size=(m, n))
    baseline_svd_partial = partial(np.linalg.svd, A, full_matrices=False)
    baseline_svd_time = time_wrapper(baseline_svd_partial, num_times)

    direct_svd_times = []
    direct_svd_eigh_times = []
    fast_svd_times = []

    # for each rank
    for q in range(2):
        direct_svd_times_q = []
        direct_svd_eigh_times_q = []

        for k in ranks:
            direct_svd_partial = partial(direct_svd, A, k=k, q=q)
            direct_svd_eigh_partial = partial(direct_svd, A, k=k, q=q, eigh=True)

            direct_svd_times_q.append(time_wrapper(direct_svd_partial, num_times))
            direct_svd_eigh_times_q.append(time_wrapper(direct_svd_eigh_partial, num_times))

        direct_svd_times.append(direct_svd_times_q)
        direct_svd_eigh_times.append(direct_svd_eigh_times_q)

    for k in ranks:
        fast_svd_partial = partial(fast_svd, A, k=k)
        fast_svd_times.append(time_wrapper(fast_svd_partial, num_times))

    return baseline_svd_time, direct_svd_times, direct_svd_eigh_times, fast_svd_times

def generate_accuracy_data(m, n, num_times, logspace_stop, ranks):
    A = np.random.normal(size=(m, n))
    U, _, Vh = np.linalg.svd(A, full_matrices=False)
    D = np.logspace(0, logspace_stop, min(m, n))
    A = np.dot(U * D, Vh)

    direct_svd_norms = []
    direct_svd_eigh_norms = []
    fast_svd_norms = []

    for q in range(2):
        direct_svd_norms_q = []
        direct_svd_eigh_norms_q = []

        for k in ranks:
            direct_svd_norms_q.append(approx_norm(A, *direct_svd(A, k=k, q=q)))
            direct_svd_eigh_norms_q.append(approx_norm(A, *direct_svd(A, k=k, q=q, eigh=True)))

        direct_svd_norms.append(direct_svd_norms_q)
        direct_svd_eigh_norms.append(direct_svd_eigh_norms_q)

    for k in ranks:
        fast_svd_norms.append(approx_norm(A, *fast_svd(A, k=k)))

    return D[ranks], direct_svd_norms, direct_svd_eigh_norms, fast_svd_norms

def plot_speed_data():
    print ("Plotting and saving speed data...")
    markers = cycle(('o', 'v', '^', '<', '>', 'x'))
    fig, ax = plt.subplots()
    ax.axhline(baseline, label="Full SVD", linestyle='--')
    [ax.plot(ranks, direct_svd[q], marker=next(markers), label="DirectSVD, q={}".format(q)) for q in range(2)]
    [ax.plot(ranks, direct_svd_eigh[q], marker=next(markers), label="EighSVD, q={}".format(q)) for q in range(2)]
    ax.plot(ranks, fast_svd, marker=next(markers), label="FastSVD")
    plt.legend()

    plt.savefig("rsvd_speed.pdf", bbox_inches="tight")

def write_out_speed_data(m, n, num_times, ranks):
    print("Generating speed data...")
    baseline, direct_svd, svd_eigh, fast_svd = generate_speed_data(m, n, num_times, ranks)

    print("Writing out CSV file for speed data...")

    with open("speed_data.csv", 'w', newline='') as f:
        speed_writer = csv.writer(f)
        speed_writer.writerow(["Ranks", "DirectSVD q=0", "DirectSVD q=1",
                                        "EighSVD q=0", "EighSVD q=1",
                                        "FastSVD", "Baseline"])
        [speed_writer.writerow([*row, baseline]) for row in zip(ranks,
                                                                direct_svd[0], direct_svd[1],
                                                                svd_eigh[0], svd_eigh[1],
                                                                fast_svd)]


def write_out_accuracy_data(m, n, num_times, ranks):
    for logspace_stop in (-0.5, -1, -2, -3.5):
        print("Generating accuracy data for logspace_stop: {}...".format(logspace_stop))
        sing_vals, direct_svd, svd_eigh, fast_svd = generate_accuracy_data(m, n, num_times, logspace_stop, ranks)

        print ("Writing out CSV file for accuracy data...")

        with open("accuracy_data_{}.csv".format(logspace_stop), 'w', newline='') as f:
            acc_writer = csv.writer(f)
            acc_writer.writerow(["Ranks", "DirectSVD q=0", "DirectSVD q=1",
                                               "EighSVD q=0", "EighSVD q=1",
                                               "FastSVD", "Baseline"])
            [acc_writer.writerow(row) for row in zip(ranks,
                                                     direct_svd[0], direct_svd[1],
                                                     svd_eigh[0], svd_eigh[1],
                                                     fast_svd, sing_vals)]


if __name__ == "__main__":
    (m, n) = (1024 * 1, 512 * 1)
    num_times = 1
    # unique is extra here but just to be sure
    ranks = np.unique(np.linspace(1, n, 10, endpoint=False, dtype=np.int32))

    write_out_speed_data(m, n, num_times, ranks)
    write_out_accuracy_data(m, n, num_times, ranks)
