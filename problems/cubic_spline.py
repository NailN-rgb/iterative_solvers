import numpy as np
from scipy.sparse import diags


def cubic_spline(x : np.array, y: np.array):
    assert(len(x) != len(y), "Input data sizes are different")
    
    n = len(x) - 1
    
    s = np.zeros(n + 1) # unknowns vector
    
    diagonals = [
        [(x[i] - x[i-1])         for i in range(1, n)],
        [2 * (x[i-1] + x[i + 1]) for i in range(1, n)],
        [(x[i + 1] - x[i])       for i in range(1, n)]
    ]
    
    # Create sparce matrix
    A = diags(diagonals, [-1, 0, 1], shape=(n+1, n+1)).to_csr()
    
    
    # set BC
    A[0, 0] = 0
    A[n, n] = 0
    
    # RHS
    b = np.zeros(n + 1)
    for i in range(1, n):
        b[i] = 6 * ((y[i+1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i-1])/(x[i] - x[i-1])) 
        
    return A, b 
    
