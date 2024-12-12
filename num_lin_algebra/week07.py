import numpy as np


def relaxation_iteration(A, b, x, w):
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

        # Subtract contributions from future variables
        for k in range(j + 1, n):
            z -= A[j, k] * y[k]

        # Update the solution for the current variable
        y[j] = w * z / A[j, j] + (1 - w) * y[j]

    return y


def relaxation(A, b, x0, eps, w):
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

    Returns:
        y : array-like, shape (n,)
            Approximate solution of the system.
        count : int
            Number of iterations performed.
    """
    x = np.copy(x0)  # Start with the initial guess
    n = len(b)
    count = 0

    while True:
        # Perform a relaxation iteration
        y = relaxation_iteration(A, b, x, w)

        # Check convergence
        error = np.linalg.norm(y - x, ord=np.inf)
        if error < eps:
            print(f"Error is {error} < {eps}. Done.")
            break

        # Update the solution and iteration count
        x = y
        count += 1

    return y, count
