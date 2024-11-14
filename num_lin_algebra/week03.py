import numpy as np


def compress(matrix: np.ndarray) -> np.ndarray:
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
        A[2][i] = A[2][i] / A[1][i]
        A[1][i + 1] = A[1][i + 1] - (A[2][i] * A[0][i])

    return A
