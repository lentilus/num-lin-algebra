import numpy as np
from numpy.testing import assert_array_almost_equal

from num_lin_algebra import relaxation, relaxation_iteration, find_optimal_w

# Define the test case for Ax = b
A = np.array(
    [
        [1 / 2, -1 / 4, 0, 0],
        [-1 / 4, 1 / 2, -1 / 4, 0],
        [0, -1 / 4, 1 / 2, -1 / 4],
        [0, 0, -1 / 4, 1 / 2],
    ],
    dtype=np.float64,
)

b = np.array([1, 2, 3, 4], dtype=np.float64)
x0 = np.array([2, 4, 6, 8], dtype=np.float64)
x_expected = np.array([16, 28, 32, 24], dtype=np.float64)

# Relaxation parameter (w) and tolerance (eps)
w = 1.0  # No relaxation, equivalent to Gauss-Seidel
eps = 1e-5  # Tolerance for the relaxation method


# Test the relaxation_iteration function
def test_relaxation_iteration():
    x = x0  # Initial guess for x

    # Define the expected result for one iteration based on the known problem
    max_iterations = 10
    expected = x_expected
    delta = np.linalg.norm(x0 - expected, ord=np.inf)

    # Loop over iterations to check if the error gets smaller
    for iteration in range(max_iterations):
        # Perform one iteration of relaxation
        x_new = relaxation_iteration(A, b, x, w)

        # Compute the error between the current solution and expected solution
        new_delta = np.linalg.norm(x_new - x_expected, ord=np.inf)
        assert new_delta < delta, f"New delta {new_delta} > {delta} (old delta)"

        print(f"Iteration {iteration + 1}: Error = {delta}")

        # Update the solution for the next iteration
        x = x_new
        delta = new_delta


# Test the relaxation method
def test_relaxation():
    solution, iterations = relaxation(A, b, x0, eps, w)

    # Test if the number of iterations is reasonable
    assert (
        iterations > 0
    ), "The relaxation method should perform at least one iteration."

    # Test if the error is within the given tolerance
    error = np.linalg.norm(solution - x_expected, ord=np.inf)

    assert error < 1e-04, f"Error in solution is too large: {error} > {eps}"


# Test the find_optimal_w function to determine the best relaxation parameter
def test_find_optimal_w():
    optimal_w, min_iterations = find_optimal_w(A, b, x0, eps)

    # Test if the function returns a valid relaxation parameter
    assert (
        optimal_w >= 0 and optimal_w <= 2
    ), "Relaxation parameter w should be in the range [0, 2]."

    # Test that the number of iterations is reasonable
    assert min_iterations > 0, "The optimal w should yield at least one iteration."

    # Test that the optimal w leads to convergence
    solution, _ = relaxation(A, b, x0, eps, optimal_w)
    error = np.linalg.norm(solution - x_expected, ord=np.inf)

    # Ensure the error is within the tolerance
    assert (
        error < 1e-04
    ), f"Error in solution with optimal w is too large: {error} > {eps}"
