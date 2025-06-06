import numpy as np
from scipy import sparse

def jacobi_solver(
    A: sparse.spmatrix, 
    b: np.ndarray,
    x0: np.ndarray,       # initial solution vector
    omega: float,         # regulirization param
    tol: float,           # absolute tolerance
    maxiter: int = 10000  # maximal iteration count
):
    n = b.shape[0] # count of unknown
    
    x = x0.astype(float).copy()
    
    # get reverse-diagonal part of A
    A_diags = A.diagonal()
    B_inv = sparse.diags(1.0 / A_diags)
    
    for iteration in range(1, maxiter):
        # current residual 
        Ax = A.dot(x)
        r = Ax - b
        
        res_norm = np.linalg.norm(r)
        
        if(res_norm < tol):
            return x, iteration - 1, res_norm
        
        x = x + omega * B_inv.dot(r)
    
    # if method does not converge return last solution approxiamtion    
    res_norm = np.linalg.norm(b - A.dot(x))
    return x, -1, res_norm
    