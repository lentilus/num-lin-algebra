import numpy as np


# We will use this as a default
def dummy_counter():
    pass


def compress(matrix: np.ndarray) -> np.ndarray:
    """
    (for testing only) encodes tridiagonal Matrix into 3xn shape
    """
    n = matrix.shape[0]
    main_diag = np.zeros(n)
    sub_diag1 = np.zeros(n)
    sub_diag2 = np.zeros(n)

    for i in range(n):
        main_diag[i] = matrix[i, i]
        if i + 1 < n:
            sub_diag1[i] = matrix[i + 1, i]
        if i - 1 >= 0:
            sub_diag2[i - 1] = matrix[i - 1, i]

    return np.array([sub_diag2, main_diag, sub_diag1])


def decompress(compressed: np.ndarray) -> np.ndarray:
    """
    (for testing only) decodes tridiagonal Matrix into 3xn shape
    """
    n = len(compressed[0])
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        matrix[i, i] = compressed[1][i]  # main
        if i + 1 < n:
            matrix[i + 1, i] = compressed[2][i]  # sub1
        if i - 1 >= 0:
            matrix[i - 1, i] = compressed[0][i - 1]  # sub2

    return matrix


def tridiag_lu(A: np.ndarray) -> np.ndarray:
    """
    Perform LU decomposition on a tridiagonal matrix stored in a compressed format in-place.

    Parameters:
    A (np.ndarray): A 3xN numpy array where:
                    - A[0, :] contains the super-diagonal entries,
                    - A[1, :] contains the main diagonal entries,
                    - A[2, :] contains the sub-diagonal entries.

    Returns:
    np.ndarray: The matrix A is modified in place to contain:
                - Super-diagonal entries for L in A[0, :],
                - Diagonal entries for U in A[1, :],
                - Sub-diagonal entries for U in A[2, :].
    """
    n = A.shape[1]

    for i in range(0, n - 1):
        pivot = A[1][i]
        # We check explicitly before dividing!
        assert not np.isclose(pivot, 0.0), "Pivot close to zero, aborted."
        A[2][i] = A[2][i] / pivot
        A[1][i + 1] = A[1][i + 1] - (A[2][i] * A[0][i])

    return A


def elim_vw(A, b):
    n = A.shape[1]
    for i in range(1, n):
        b[i] = b[i] - (A[2][i - 1] * b[i - 1])


def elim_rw(A, b):
    n = A.shape[1]
    b[n - 1] = b[n - 1] / A[1][n - 1]
    for i in range(1, n):
        j = n - i - 1
        b[j] = (b[j] - (A[0][j] * b[j + 1])) / A[1][j]


def tridiag_vwrw(Z, b):
    # tridiag modifies Z in place so no need to capture return
    tridiag_lu(Z)
    elim_vw(Z, b)
    elim_rw(Z, b)
    return b
