import numpy as np
from scipy.sparse import rand, csr_matrix
from scipy.sparse import diags

def generate_spd_matrix(n, density=0.1, random_state=None):
    """
    Идея в генерации разреженных SPD матриц 
    Положим, что такая матрица будет служить приближением к матрицам получаемым для FEM
    
    """
    A = rand(n, n, density=density, format='csr', random_state=random_state)
    
    A = (A + A.T) / 2

    # Обеспечение положительной определенности:
    # A_spd = A.T @ A + alpha * I
    A = A @ A.T

    # Обеспечение диагонального преобладания
    alpha = n * 1e-3
    I = diags([alpha] * n, 0)
    A = A + I

    return A