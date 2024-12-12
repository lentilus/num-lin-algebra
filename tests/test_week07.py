import numpy as np
from numpy._core.multiarray import dtype
from numpy.testing import assert_array_almost_equal

from num_lin_algebra import relaxation, relaxation_iteration, find_optimal_w

""" Test Case (i) """
A1 = np.array(
    [
        [1 / 2, -1 / 4, 0, 0],
        [-1 / 4, 1 / 2, -1 / 4, 0],
        [0, -1 / 4, 1 / 2, -1 / 4],
        [0, 0, -1 / 4, 1 / 2],
    ],
    dtype=np.float64,
)

b1 = np.array([1, 2, 3, 4], dtype=np.float64)
expected1 = np.array([16, 28, 32, 24], dtype=np.float64)
eps1 = 1e-5

""" Test Case (ii) """

h = 0.01
n = 50

# Build A2
A2 = np.ones((n, n), dtype=np.float64)  # Start with a matrix of one
A2 -= np.eye(n)  # Subtract the identity matrix
A2 *= -h  # Scale the off-diagonal elements by -h
np.fill_diagonal(A2, 1)  # Set the diagonal elements to 1

# Adjust the lower triangular part to have h instead of -h
for i in range(1, n):
    for j in range(i):
        A2[i, j] = h

b2 = np.ones(n, dtype=np.float64)
eps2 = 1e-6


# Test the relaxation_iteration function
def test_relaxation_iteration():
    max_iterations = 10
    expected = expected1
    delta = np.linalg.norm(b1 - expected, ord=np.inf)

    # start with b itself as guess
    x = b1

    # Loop over iterations to check if the error gets smaller
    for iteration in range(max_iterations):
        # Perform one iteration of relaxation
        x_new = relaxation_iteration(A1, b1, x, w=1.0)

        # Compute the error between the current solution and expected solution
        new_delta = np.linalg.norm(x_new - expected1, ord=np.inf)
        assert new_delta < delta, f"New delta {new_delta} > {delta} (old delta)"

        print(f"Iteration {iteration + 1}: Error = {delta}")

        # Update the solution for the next iteration
        x = x_new
        delta = new_delta


# Test the relaxation method
def test_relaxation():
    solution, iterations = relaxation(A1, b1, b1, eps1, w=1.0)

    # Test if the number of iterations is reasonable
    assert (
        iterations > 0
    ), "The relaxation method should perform at least one iteration."

    # Test if the error is within the given tolerance
    error = np.linalg.norm(solution - expected1, ord=np.inf)

    assert error < 1e-04, f"Error in solution is too large: {error} > {eps1}"


# Test the find_optimal_w function to determine the best relaxation parameter
def test_case_i():
    optimal_w, min_iterations = find_optimal_w(A1, b1, b1, 10e-5)

    # Test if the function returns a valid relaxation parameter
    assert (
        optimal_w >= 0 and optimal_w <= 2
    ), "Relaxation parameter w should be in the range [0, 2]."

    # Test that the number of iterations is reasonable
    assert min_iterations > 0, "The optimal w should yield at least one iteration."

    # Test that the optimal w leads to convergence
    solution, _ = relaxation(A1, b1, b1, eps1, optimal_w)
    error = np.linalg.norm(solution - expected1, ord=np.inf)

    # Ensure the error reasonable
    assert (
        error < 1e-04
    ), f"Error in solution with optimal w is too large: {error} > {eps1}"


def test_case_ii():
    optimal_w, min_iterations = find_optimal_w(A2, b2, b2, eps2)
    # Test if the function returns a valid relaxation parameter
    assert (
        optimal_w >= 0 and optimal_w <= 2
    ), "Relaxation parameter w should be in the range [0, 2]."

    # Test that the number of iterations is reasonable
    assert min_iterations > 0, "The optimal w should yield at least one iteration."

    # Test that the optimal w leads to convergence
    solution, _ = relaxation(A2, b2, b2, eps2, optimal_w)
    print(f"The solution is {solution}.")
