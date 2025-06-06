import numpy as np
from scipy.sparse import diags


# Generate matrix for cubic spline second derivatives
# Method get only matrix size. Spline approximate sin(x) function
 
def cubic_spline(n, x = (0, 1), seed=None):
    
    if seed is None:
        np.random.seed(seed)
        
    x = np.linspace(*x, n + 1)
    y = np.sin(x) + 0.1 * np.random.randn(n + 1)  # добавим шум
    
    n = len(x) - 1
    
    s = np.zeros(n + 1) # unknowns vector
    
    diagonals = [
        [(x[i] - x[i-1])         for i in range(1, n)],
        [2 * (x[i-1] + x[i + 1]) for i in range(1, n)],
        [(x[i + 1] - x[i])       for i in range(1, n)]
    ]
    
    # Create sparce matrix
    A = diags(diagonals, [-1, 0, 1], shape=(n-1, n-1)).tocsr()
    
    # RHS
    b = np.zeros(n - 1)
    for i in range(1, n):
        b[i - 1] = 6 * ((y[i+1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i-1])/(x[i] - x[i-1])) 
        
        
    return A, b 
    
