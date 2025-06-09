import numpy as np
from scipy import sparse

def pcg(
    A: sparse.spmatrix, 
    b: np.ndarray, 
    x0: np.ndarray, 
    tol: float = 1e-6
):
    n = A.shape[0]
    x = x0.copy()
    r = b - A @ x
    
    # Диагональный предобусловливатель
    M_inv = 1 / A.diagonal()  # M^{-1} = diag(1/a_ii)
    z = M_inv * r  # Решаем Mz = r
    p = z.copy()

    rz_old = r @ z

    for i in range(n - 1):
        res_norm = np.linalg.norm(r)
        if res_norm < tol:
            return x, i, res_norm

        Ap = A @ p
        alpha = rz_old / (p @ Ap)

        x = x + alpha * p
        r = r - alpha * Ap

        z = M_inv * r  # Решаем Mz = r
        rz_new = r @ z
        beta = rz_new / rz_old

        p = z + beta * p
        rz_old = rz_new

    return x, -1, np.linalg.norm(r)