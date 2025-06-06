import numpy as np
from scipy.sparse import diags


def laplase_problem_generator(n):
    f = 10 # rhs const function
    
    h = 1 / n # mesh step
    
    matrix_diags = [
        [(-1) for i in range(n)],
        [(2)  for i in range(n)],
        [(-1) for i in range(n)]
    ]
    
    A = diags(matrix_diags, [-1, 0, 1], shape=(n - 1, n - 1)).tocsr()
    
    # # set BC
    # A[0, 0] = 1
    # A[n - 2, n - 2] = 1
    
     # RHS
    b = np.zeros(n - 1)
    for i in range(1, n - 1):
        b[i] = f
        
    b[0] = 1
    b[n - 2] = 1
    
    return A, b
