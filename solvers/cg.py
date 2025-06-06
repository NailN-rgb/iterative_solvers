import numpy as np
from scipy import sparse

def c_gradient(
    A: sparse.spmatrix,
    b: np.ndarray,
    x0: np.ndarray,
    tol: float = 1e-6 
):
    n = b.shape[0] # unknowns count
    x = x0.copy()
    
    # initial residual
    r = b - A.dot(x)
    # initial p0 vector = r0
    p = r.copy()
    
    residual_skalar_initial = r.dot(r)
    
    for i in range(1, n):
        
        if r < tol:
            return x, i, r
        
        Ap = A.dot(p)
        
        alpha = residual_skalar_initial / Ap.dot(p)
        
        x = x + alpha * p
        
        r_new = r - alpha * Ap
        
        beta = (r_new.dot(r_new)) / (r.dot(r))
        
        r = r_new
        
        p = r + beta * p
        
    # if method does not converge
    return x, -1, r
    