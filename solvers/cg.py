import numpy as np
from scipy import sparse

def c_gradient(
    A: sparse.spmatrix,
    b: np.ndarray,
    x0: np.ndarray,
    tol: float = 1e-6 
):
    n = A.shape[0]
    x = x0.copy()
    r = b - A @ x
    p = r.copy()

    rs_old = r @ r

    for i in range(n - 1):
        # Проверка сходимости
        res_norm = np.linalg.norm(r)
        if res_norm < tol:
            return x, i, res_norm

        Ap = A @ p
        alpha = rs_old / (p @ Ap)

        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = r @ r
        beta = rs_new / rs_old

        p = r + beta * p
        rs_old = rs_new
        
        # print(res_norm)
        
    # if method does not converge
    return x, -1, np.linalg.norm(r)
    