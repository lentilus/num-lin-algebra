import numpy as np


def dummy_counter():
    """
    A function that does nothing
    """
    pass


def new_counter():
    """
    returns a closure that counts
    """
    counter = 0

    def increment():
        nonlocal counter
        counter += 1
        return counter

    return increment


class NoConvergenceError(Exception):
    """
    A custom exception for cases where the algorithm does not converge
    """

    def __init__(self, message="Algorithm does not converge"):
        super().__init__(message)


def relaxation_iteration(A, b, x, w, counter=dummy_counter):
    """
    Perform one iteration step of the relaxation method.

    Parameters:
        A : array-like, shape (n, n)
            Coefficient matrix of the system.
        b : array-like, shape (n,)
            Right-hand side vector of the linear system.
        x : array-like, shape (n,)
            Previously computed solution vector.
        w : float
            Relaxation parameter.

    Returns:
        y : array-like, shape (n,)
            New solution vector after one iteration step.
    """
    n = len(b)
    y = np.copy(x)  # Initialize the new solution vector with the old values

    for j in range(n):
        z = b[j]

        # Subtract contributions from previous variables
        for k in range(j):
            z -= A[j, k] * y[k]
            counter()

        # Subtract contributions from future variables
        for k in range(j + 1, n):
            z -= A[j, k] * y[k]
            counter()

        # Update the solution for the current variable
        y[j] = w * z / A[j, j] + (1 - w) * y[j]
        counter()

    return y


def relaxation(A, b, x0, eps, w, counter=dummy_counter):
    """
    Solve a linear system using the Gauss-Seidel relaxation method.

    Parameters:
        A : array-like, shape (n, n)
            Coefficient matrix of the system.
        b : array-like, shape (n,)
            Right-hand side vector of the linear system.
        x0 : array-like, shape (n,)
            Initial guess for the solution.
        eps : float
            Tolerance parameter. Iteration stops when the maximum norm of the
            difference between consecutive solutions is less than eps.
        w : float
            Relaxation parameter.
        max_no_progress_iter : int, optional (default is 100)
            Maximum number of iterations without improvement in error before throwing an exception.

    Returns:
        y : array-like, shape (n,)
            Approximate solution of the system.
        count : int
            Number of iterations performed.

    Raises:
        Exception: If no progress is made in the specified number of iterations.
    """
    x = np.copy(x0)  # Start with the initial guess
    count = 0
    prev_error = np.inf  # Initialize previous error as infinity

    while True:
        y = relaxation_iteration(A, b, x, w, counter=counter)

        # Compute the error (change in solution)
        error = np.linalg.norm(y - x, ord=np.inf)

        # Check if the error decreases
        if error >= prev_error:
            raise NoConvergenceError

        prev_error = error  # Update the previous error if progress is made

        # Check for convergence break condition
        if error < eps:
            break

        # Update the solution and iteration count
        x = y
        count += 1

    return y, count


def find_optimal_w(
    A,
    b,
    x0,
    eps,
    w_min=0.1,
    w_max=2.0,
    w_step=0.001,
):
    optimal_w = w_min
    min_iterations = float("inf")
    min_operations = -1

    # Iterate over a range of w values
    for w in np.arange(w_min, w_max + w_step, w_step):
        # print(f"Computing w={w}")
        try:
            # create a new counter for the arithmetic operations
            counter = new_counter()

            # calculte the number of iterations
            _, iterations = relaxation(A, b, x0, eps, w, counter=counter)
        except NoConvergenceError:
            # print(f"No convergence for w={w}")
            continue
        if iterations < min_iterations:
            min_iterations = iterations
            optimal_w = w
            min_operations = counter() - 1

    print(
        f"Optimal w: {optimal_w}, with {min_iterations} iterations and {min_operations} flops."
    )
    return optimal_w, min_iterations
