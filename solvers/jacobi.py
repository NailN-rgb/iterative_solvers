import numpy as np
from scipy import sparse

def jacobi_solver(
    A: sparse.spmatrix, 
    b: np.ndarray,
    x0: np.ndarray,       # initial solution vector
    tol: float = 1e-6,           # absolute tolerance
    maxiter: int = 10000  # maximal iteration count
):
    n = b.shape[0]
    x = x0.astype(float).copy()

    D_inv = 1.0 / A.diagonal()
    
    # Предварительно A - D (L + U)
    R = A.copy()
    R.setdiag(0)

    for iteration in range(maxiter):
        x_new = D_inv * (b - R.dot(x))
        res_norm = np.linalg.norm(x_new - x)
        if res_norm < tol:
            return x_new, iteration, res_norm
        x = x_new
    
    # if method does not converge return last solution approxiamtion    
    res_norm = np.linalg.norm(b - A.dot(x))
    return x, -1, res_norm
    