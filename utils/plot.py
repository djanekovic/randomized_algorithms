import numpy as np
from matplotlib import pyplot as plt


def plot_singular_values(A):
    fig, ax = plt.subplots()
    d = np.linalg.svd(A, full_matrices=False, compute_uv=False)
    ax.scatter(list(range(1, len(d)+1)), d)

    return fig, ax

def plot_matrix(A):
    fig, ax = plt.subplots()
    ax.matshow(A)

    return fig, ax


if __name__ == "__main__":
    import utils

    A = utils.generate_random_symmetric_matrix(100)
    fig, ax = plot_singular_values(A)

    plt.show()
