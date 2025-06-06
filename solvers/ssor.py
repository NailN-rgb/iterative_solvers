import numpy as np
from scipy import sparse

def ssor_solver(
    A: sparse.spmatrix, 
    b: np.ndarray,
    x0: np.ndarray,       # initial solution vector
    omega: float,         # regulirization param
    tol: float,           # absolute tolerance
    maxiter: int = 10000  # maximal iteration count
):
    n = b.shape[0] # count of unknown
    
    x = x0.astype(float).copy()
    x_half = np.zeros(n)
    
    for iteration in range(1, maxiter):
        # current residual 
        Ax = A.dot(x)
        r = Ax - b
        
        x_old = x.copy()
        
        res_norm = np.linalg.norm(r)
        
        if(res_norm < tol):
            return x, iteration - 1, res_norm
        
        # staight correction
        for i in range(n):
            row_first = A.indptr[i]
            row_last  = A.indptr[i + 1]
            Aidx = A.indeces[row_first:row_last]
            Avals = A.data[row_first:row_last]
            
            
            a_ii = 0.0
            sigma = 0.0

            for idx, j in enumerate(Aidx):
                if j == i:
                    a_ii = Avals[idx]
                elif j < i:
                    sigma += Avals[idx] * x_half[j]  # updated solution value
                else:
                    sigma += Avals[idx] * x_old[j]  # old solution value

            x_half[i] = (1 - omega) * x_old[i] + (omega / a_ii) * (b[i] - sigma)
        
        
        # backward correction
        for i in range(n):
            row_first = A.indptr[i]
            row_last  = A.indptr[i + 1]
            Aidx = A.indeces[row_first:row_last]
            Avals = A.data[row_first:row_last]
            
            
            a_ii = 0.0
            sigma = 0.0

            for idx, j in enumerate(Aidx):
                if j == i:
                    a_ii = Avals[idx]
                elif j < i:
                    sigma += Avals[idx] * x[j]  # updated solution value
                else:
                    sigma += Avals[idx] * x_half[j]  # old solution value

            x[i] = (1 - omega) * x_half[i] + (omega / a_ii) * (b[i] - sigma)
            
            
            
    # if method does not converge return last solution approxiamtion    
    res_norm = np.linalg.norm(b - A.dot(x))
    return x, -1, res_norm
    