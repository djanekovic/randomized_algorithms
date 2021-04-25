import numpy as np
import utils

def gpu_rsvd(A, k=0, p=10, q=0, block_size=10):
    m, n = A.shape

    if k == 0:
        k = min(m, n)

    # form n x (k+p) Gaussian random matrix G
    G = np.random.normal(size=(n, k+p))

    # number of columns we are pushing in the pipeline
    col_num = 2
    current_column = 0

    Y = np.zeros((m, k+p))

    while current_column < n:
        a = A[:, current_column:current_column + col_num]
        g = G[current_column:current_column + col_num]

        y = a @ g

        if q == 0:
            Y += y

        for _ in range(q):
            Y += np.outer(a, a) @ y

        current_column += col_num

    # Here we have all the columns and Y should be A @ G
    assert np.allclose(Y, A @ G)


    Q, _ = np.linalg.qr(Y, mode='reduced')


    current_column = 0
    B = np.empty((k+p, n))
    while current_column < n:
        a = A[:, current_column:current_column + col_num]

        B[:, current_column:current_column + col_num] = Q.T @ a

        current_column += col_num

    assert np.allclose(B, Q.T @ A)

    # Form the SVD of the small matrix
    Uhat, d, vh = np.linalg.svd(B, full_matrices=False)

    u = Q @ Uhat

    return u, d, vh


def rsvd_basic(A, k=0, p=10, q=0):
    """Basic randomized SVD

    Input is mxn matrix A, a target rank k and over-sampling parameter p. Default rank k is
    min(m, n).

    We return:
        U: left singular vectors
        D: diagonal matrix with singular values
        V*: right singular vectors
    """

    m, n = A.shape

    if k == 0:
        k = min(m, n)

    # form n x (k+p) Gaussian random matrix G
    G = np.random.normal(size=(n, k+p))

    # Form the sample matrix m x (k+p)
    Y = A @ G

    for _ in range(q):
        Z = A.conj().T @ Y
        Y = A @ Z

    # Orhonormalize the columns
    # Q is mx(k+p)
    Q, _ = np.linalg.qr(Y, 'reduced')

    # Form the (k+p) x n matrix
    B = Q.conj().T @ A

    # Form the SVD of the small matrix
    Uhat, d, vh = np.linalg.svd(B, full_matrices=False)

    u = Q @ Uhat

    return u, d, vh


def rsvd_basic_accuracy(A, k=0, p=10, q=0):
    """Randomized SVD with improved accuracy

    Input is mxn matrix A, a target rank k and over-sampling parameter p. Default rank k is
    min(m, n).

    We return:
        U: left singular vectors
        D: diagonal matrix with singular values
        V*: right singular vectors
    """

    m, n = A.shape

    if k == 0:
        k = min(m, n)

    # form n x (k+p) Gaussian random matrix G
    G = np.random.normal(size=(n, k+p))

    # Form the sample matrix m x (k+p)
    Y = A @ G

    # Orhonormalize the columns
    # Q is mx(k+p)
    Q, _ = np.linalg.qr(Y, 'reduced')

    for _ in range(q):
        W, _ = np.linalg.qr(A.conj().T @ Q, 'reduced')
        Q, _ = np.linalg.qr(A @ W, 'reduced')

    # Form the (k+p) x n matrix
    B = Q.conj().T @ A

    # Form the SVD of the small matrix
    Uhat, d, vh = np.linalg.svd(B, full_matrices=False)

    u = Q @ Uhat

    return u, d, vh


def single_pass_rsvd(A, k=0, p=10):
    """Single pass rsvd algorithm

    For now we assert that A is symmetric, we will relax later. This method does
    inplace A = U D U* decomposition.

    We return:
        U: orthonormal matrix
        D: diagonal matrix with eigenvalues
    """

    assert np.allclose(A, A.conj().T)

    # A is nxn matrix
    n, _ = A.shape

    # form n x (k+p) Gaussian random matrix G
    G = np.random.normal(size=(n, k+p))

    # Form the sample matrix, this can be inplace since we don't need A anymore
    # but since there is no inplace gemm this is not very useful.
    Y = A @ G

    # Form QR factorization of Y -> n x n and  n x (k+p)
    # We can also generate partial QR: n x (k+p) and (k+p) x (k+p)
    Q, _ = np.linalg.qr(Y, 'reduced')

    # Solve least squares problem C (Q*G) = Q*Y for C
    #
    # Attack everything with conjugate transpose:
    # (G*Q) C* = Y*Q
    Ch, res, _, _ = np.linalg.lstsq(G.conj().T @ Q, Y.conj().T @ Q)

    w, v = np.linalg.eigh(Ch)

    return Q @ v, w


def test_rsvd_basic_cmp_util(n, m, k, p=0, l=0, q=0):
    A = utils.generate_random_matrix(n, m, k=k)

    # first k should be the same
    u, d, vh = rsvd_basic(A, k=k-l, p=p, q=q)
    A_ = (u @ np.diag(d)) @ vh

    #u_, d_, vh_ = np.linalg.svd(A, full_matrices=False)
    # assert np.allclose(u[:, :k], u_[:, :k])
    # assert np.allclose(d[:k], d_[:k])
    # assert np.allclose(vh[:k], vh_[:k, :])

    A_norm = np.linalg.norm(A)
    print("RSVD norm:", np.linalg.norm(A - A_)/A_norm)

def test_rsvd_basic():
    test_rsvd_basic_cmp_util(100, 50, 50)

    # Matrix is 100x50 and rank 40
    # We set oversamping param to the 9 and we ask for rank 30 approximation
    test_rsvd_basic_cmp_util(100, 50, 40, p=9, l=10)
    # same thing but we are doing power iteration with q=2
    test_rsvd_basic_cmp_util(100, 50, 40, p=9, l=10, q=2)

    test_rsvd_basic_cmp_util(100, 50, 30, p=9, l=10)
    test_rsvd_basic_cmp_util(100, 50, 30, p=9, l=10, q=2)

    test_rsvd_basic_cmp_util(100, 50, 20, p=9, l=10)
    test_rsvd_basic_cmp_util(100, 50, 20, p=9, l=10, q=2)

    test_rsvd_basic_cmp_util(100, 50, 50, p=5, l=5)
    test_rsvd_basic_cmp_util(100, 50, 20, p=5, l=5)
    test_rsvd_basic_cmp_util(100, 50, 10, p=5, l=5)
    test_rsvd_basic_cmp_util(100, 50, 5,  p=5, l=5)

    test_rsvd_basic_cmp_util(100, 50, 50, p=1)
    test_rsvd_basic_cmp_util(100, 50, 20, p=1)
    test_rsvd_basic_cmp_util(100, 50, 10, p=1)
    test_rsvd_basic_cmp_util(100, 50, 5, p=1)

if __name__ == "__main__":
    test_rsvd_basic()

    A = utils.generate_random_matrix(10, 5)

    u, d, vh = rsvd_basic(A)
    u_, d_, vh_ = np.linalg.svd(A, full_matrices=False)

    np.allclose(u, u_)
    np.allclose(d, d_)
    np.allclose(vh, vh_)

    A = utils.generate_random_symmetric_matrix(10, 5)
    u, d = single_pass_rsvd(A, 5, 5)

    u_, d_ = np.linalg.eigh(A)
    np.allclose(u, u_)
    np.allclose(d, d_)

    A = utils.generate_random_matrix(20, 10, 5)
    u, d, vh = rsvd_basic(A, k=45, p=0, q=2)
    A_ = u @ np.diag(d) @ vh
    print(np.linalg.norm(A - A_))

    gpu_rsvd(A)
