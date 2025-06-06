import numpy as np
from scipy import sparse

from scipy.sparse.linalg import eigsh


def simple_iteration_solver_L_precond(
    A: sparse.spmatrix, 
    b: np.ndarray,
    x0: np.ndarray,       # initial solution vector
    omega: float = 1,         # regulirization param
    tol: float = 1e-6,           # absolute tolerance
    maxiter: int = 10000  # maximal iteration count
):
    # Not optimal
    lambda_max = eigsh(A, k=1, which='LA', return_eigenvectors=False)[0]
    lambda_min = eigsh(A, k=1, which='SA', return_eigenvectors=False)[0]
    tau = 2.0 / (lambda_min + lambda_max)
    
    M = np.tril(A.toarray())
        
    x = x0.astype(float).copy()
    
    for iteration in range(maxiter):
        r = A.dot(x) - b
        z = np.linalg.solve(M, r)
        res_norm = np.linalg.norm(r)

        if res_norm < tol:
            return x, iteration, res_norm

        x = x - tau * z
    
    # if method does not converge return last solution approxiamtion    
    res_norm = np.linalg.norm(b - A.dot(x))
    return x, -1, res_norm
    