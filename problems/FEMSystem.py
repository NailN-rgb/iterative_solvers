import numpy as np
from scipy.sparse import diags


def laplase_problem_generator(x):
    f = 10 # rhs const function
    
    n = len(x) - 1
    h = np.diff(x) # mesh step
    
    matrix_diags = [
        [(-1 / h[i]) for i in range(1, n)],
        [(2 / h[i])  for i in range(1, n)],
        [(-1 / h[i]) for i in range(1, n)]
    ]
    
    A = diags(matrix_diags, [-1, 0, 1], shape=(n+1, n+1)).to_csr()
    
    # set BC
    A[0, 0] = 1
    A[n, n] = 1
    
     # RHS
    b = np.zeros(n + 1)
    for i in range(1, n):
        b[i] = h[i] / 2 * f
        
    b[0] = 1
    b[n] = 1
    
    return A, b
