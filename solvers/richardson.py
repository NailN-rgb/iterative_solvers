import numpy as np
from scipy import sparse

from scipy.sparse.linalg import eigsh


def simple_iteration_solver(
    A: sparse.spmatrix, 
    b: np.ndarray,
    x0: np.ndarray,       # initial solution vector
    omega: float = 1,         # regulirization param
    tol: float = 1e-6,           # absolute tolerance
    maxiter: int = 10000  # maximal iteration count
):
    # Not optimal
    x = x0.astype(float).copy()
    
    for iteration in range(maxiter):
        r = A.dot(x) - b
        res_norm = np.linalg.norm(r)

        if res_norm < tol:
            return x, iteration, res_norm

        x = x - omega * r
    
    # if method does not converge return last solution approxiamtion    
    res_norm = np.linalg.norm(b - A.dot(x))
    return x, -1, res_norm
    