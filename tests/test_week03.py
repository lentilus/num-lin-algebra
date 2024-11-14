import numpy as np
import pytest
from num_lin_algebra import *


def test_compress():
    matrix = np.array([[1, 2, 0, 0], [3, 4, 5, 0], [0, 6, 7, 8], [0, 0, 9, 10]])
    compressed = compress(matrix)
    expected = np.array(
        [
            [2, 5, 8, 0],
            [1, 4, 7, 10],
            [3, 6, 9, 0],
        ]
    )
    np.testing.assert_array_equal(compressed, expected)


def test_decompress():
    compressed = np.array(
        [
            [0, 3, 6, 0],
            [1, 4, 7, 10],
            [2, 5, 8, 0],
        ]
    )
    decompressed = decompress(compressed)
    expected_decompressed = np.array(
        [[1, 0, 0, 0], [2, 4, 3, 0], [0, 5, 7, 6], [0, 0, 8, 10]]
    )
    np.testing.assert_array_equal(decompressed, expected_decompressed)


def test_round_trip():
    matin = np.array([[100, 0, 0, 0], [35, 0, 2, 0], [0, 3, -12, 80], [0, 0, 99, 13]])

    np.testing.assert_equal(decompress(compress(matin)), matin)


def test_tridiag_lu():
    input = np.array([[1, 0, 0], [2, -2, 3], [0, 4, 2]])
    lu = np.array([[1, 0, 0], [2, -2, 3], [0, -2, 8]])

    np.testing.assert_equal(decompress(tridiag_lu(compress(input))), lu)


def test_elim_vw():
    lu = np.array([[1, 0, 0], [2, -2, 3], [0, -2, 8]])
    b = np.array([1, -3, 2])
    x = np.array([1, -5, -8])

    elim_vw(compress(lu), b)
    np.testing.assert_equal(b, x)


def test_elim_rw():
    lu = np.array([[1, 0, 0], [2, -2, 3], [0, -2, 8]])
    b = np.array([0.0, 2.0, 1.0])
    x = np.array([0, -13.0 / 16.0, 1.0 / 8.0])

    elim_rw(compress(lu), b)
    np.testing.assert_equal(b, x)


""" Offizielle Evaluierung der Algorithmen """


def test_official1():
    A = np.array(
        [
            [10.0, 1.0, 0.0, 0.0],
            [4.0, 20.0, 2.0, 0.0],
            [0.0, 5.0, 30.0, 3.0],
            [0.0, 0.0, 6.0, 40.0],
        ]
    )
    b = np.array([12.0, 50.0, 112.0, 178.0])
    matin = compress(A)
    solution = np.array([1.0, 2.0, 3.0, 4.0])
    tridiag_vwrw(matin, b)
    # finishes with matin = [1.0,2.0,3.0,4.0]
    np.testing.assert_almost_equal(b, solution)
