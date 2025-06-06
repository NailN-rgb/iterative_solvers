import numpy as np
from scipy import sparse

def simple_iteration_solver(
    A: sparse.spmatrix, 
    b: np.ndarray,
    x0: np.ndarray,       # initial solution vector
    omega: float,         # regulirization param
    tol: float,           # absolute tolerance
    maxiter: int = 10000  # maximal iteration count
):
    n = b.shape[0] # count of unknown
    
    x = x0.astype(float).copy()
    
    for iteration in range(1, maxiter):
        # current residual 
        Ax = A.dot(x)
        r = Ax - b
        
        res_norm = np.linalg.norm(r)
        
        if(res_norm < tol):
            return x, iteration - 1, res_norm
        
        x = x + omega * r
    
    # if method does not converge return last solution approxiamtion    
    res_norm = np.linalg.norm(b - A.dot(x))
    return x, -1, res_norm
    