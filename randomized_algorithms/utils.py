import numpy as np


def generate_random_symmetric_matrix(n, k=0):
    """ Generate random nxn matrix with rank k

    We return S = A @ A.T

    Returns np.matrix
    """

    matrix = generate_random_matrix(n, n, k)

    return matrix @ matrix.T


def generate_random_matrix(m, n, k=0):
    """ Generate random mxn matrix with rank k.
    Returns np.matrix
    """

    matrix = np.random.normal(size=(m, n))
    if k:
        return truncated_svd(matrix, k)

    return matrix

def truncated_svd(A, rank):
    u, d, vh = np.linalg.svd(A, full_matrices=False)
    d[rank:] = 0

    return (u @ np.diag(d)) @ vh


def test_generate_random_matrix():
    # Generate 10x5 matrix with rank 5
    A = generate_random_matrix(10, 5)
    assert A.shape == (10, 5)
    assert np.linalg.matrix_rank(A) == 5

    # Generate 10x5 matrix with rank 3
    A = generate_random_matrix(10, 5, k=3)
    assert A.shape == (10, 5)
    assert np.linalg.matrix_rank(A) == 3

    # Generate 5x10 matrix with rank 5
    A = generate_random_matrix(5, 10)
    assert A.shape == (5, 10)
    assert np.linalg.matrix_rank(A) == 5

    # Generate 5x10 matrix with rank 3
    A = generate_random_matrix(5, 10, k=3)
    assert A.shape == (5, 10)
    assert np.linalg.matrix_rank(A) == 3

def test_generate_random_symmetric_matrix():
    # Generate 10x10 matrix with rank 10
    A = generate_random_symmetric_matrix(10)
    assert A.shape == (10, 10)
    assert np.allclose(A.T, A)
    assert np.linalg.matrix_rank(A) == 10

    # Generate 10x10 matrix with rank 5
    A = generate_random_symmetric_matrix(10, k=5)
    assert A.shape == (10, 10)
    assert np.allclose(A.T, A)
    assert np.linalg.matrix_rank(A) == 5


if __name__ == "__main__":
    test_generate_random_matrix()
    test_generate_random_symmetric_matrix()
