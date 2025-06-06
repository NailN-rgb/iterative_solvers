import numpy as np
from scipy import sparse

def sor_solver(
    A: sparse.spmatrix, 
    b: np.ndarray,
    x0: np.ndarray,       # initial solution vector
    omega: float = 1.,         # regulirization param
    tol: float = 1e-6,           # absolute tolerance
    maxiter: int = 10000  # maximal iteration count
):
    n = b.shape[0] # count of unknown
    
    x = x0.astype(float).copy()
    
    for iteration in range(maxiter):
        x_old = x.copy()
        
        for i in range(n):
            row_first = A.indptr[i]
            row_last  = A.indptr[i + 1]
            Aidx = A.indices[row_first:row_last]
            Avals = A.data[row_first:row_last]
        
            a_ii = 0.0
            sigma = 0.0

            for idx, j in enumerate(Aidx):
                if j == i:
                    a_ii = Avals[idx]
                elif j < i:
                    sigma += Avals[idx] * x[j]  # updated solution value
                else:
                    sigma += Avals[idx] * x_old[j]  # old solution value

            x[i] = (1 - omega) * x_old[i] + (omega / a_ii) * (b[i] - sigma)
            
        # current residual 
        Ax = A.dot(x)
        r = Ax - b
        
        res_norm = np.linalg.norm(r)
        
        if(res_norm < tol):
            return x, iteration - 1, res_norm
        
            
    # if method does not converge return last solution approxiamtion    
    res_norm = np.linalg.norm(b - A.dot(x))
    return x, -1, res_norm
    