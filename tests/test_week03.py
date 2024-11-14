import numpy as np
from num_lin_algebra import *


# Test for compress function
def test_compress():
    # Test case 1: Basic tridiagonal matrix (4x4)
    matrix_1 = np.array([[1, 2, 0, 0], [3, 4, 5, 0], [0, 6, 7, 8], [0, 0, 9, 10]])
    compressed_1 = compress(matrix_1)
    expected_compressed_1 = np.array(
        [
            [2, 5, 8, 0],  # sub_diag2 (above main diagonal)
            [1, 4, 7, 10],  # main_diag
            [3, 6, 9, 0],  # sub_diag1 (below main diagonal)
        ]
    )
    np.testing.assert_array_equal(compressed_1, expected_compressed_1)


# Test for decompress function
def test_decompress():
    # Test case 1: Compressed form of a 4x4 matrix
    compressed_1 = np.array(
        [
            [0, 3, 6, 0],
            [1, 4, 7, 10],
            [2, 5, 8, 0],
        ]
    )
    decompressed_1 = decompress(compressed_1)
    expected_decompressed_1 = np.array(
        [[1, 0, 0, 0], [2, 4, 3, 0], [0, 5, 7, 6], [0, 0, 8, 10]]
    )
    np.testing.assert_array_equal(decompressed_1, expected_decompressed_1)


def test_round_trip():
    matin = np.array([[100, 0, 0, 0], [35, 0, 2, 0], [0, 3, -12, 80], [0, 0, 99, 13]])

    np.testing.assert_equal(decompress(compress(matin)), matin)


def test_tridiag_lu():
    input = np.array([[1, 0, 0], [2, -2, 3], [0, 4, 2]])
    lu = np.array([[1, 0, 0], [2, -2, 3], [0, -2, 8]])

    np.testing.assert_equal(decompress(tridiag_lu(compress(input))), lu)
