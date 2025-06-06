from scipy import sparse
import numpy as np

from solvers.jacobi import jacobi_solver


def ruge_stuben_variables(
    A: sparse.spmatrix, 
    theta: float
):
    # get coarse and fine variables at matrix
    n = A.shape[0]
    coarse = set()
    fine   = set()
    
    strong_connections = [set() for _ in range(n)]
    for i in range(n):
        row_start = A.indptr[i]
        row_end = A.indptr[i + 1]
        max_offdiag = np.max(np.abs(A.data[row_start:row_end][A.indices[row_start:row_end] != i]))
        if max_offdiag == 0:
            continue
        for j in range(row_start, row_end):
            col = A.indices[j]
            if i != col and abs(A.data[j]) >= theta * max_offdiag:
                strong_connections[i].add(col)
                strong_connections[col].add(i)
    
    # Выбор C/F переменных
    undecided = set(range(n))
    while undecided:
        max_connections = -1
        max_node = None
        for i in undecided:
            count = len(strong_connections[i] & undecided)
            if count > max_connections:
                max_connections = count
                max_node = i
        if max_node is None:
            fine.update(undecided)
            break
        coarse.add(max_node)
        undecided.remove(max_node)
        for j in strong_connections[max_node]:
            if j in undecided:
                fine.add(j)
                undecided.remove(j)
    
    return list(coarse), list(fine)



def build_interpolator(
    A: sparse.spmatrix,
    C: list,
    F: list
):
    n = A.shape[0]
    n_c = len(C)
    C_index = {c: i for i, c in enumerate(C)}
    row = []
    col = []
    data = []
    
    # добавляем сильные связи
    for i, c in enumerate(C):
        row.append(c)
        col.append(i)
        data.append(1.0)
    
    a_diag = A.diagonal()
    
    # для слабых связей
    for f in F:
        row_start = A.indptr[f]
        row_end = A.indptr[f + 1]
        strong_C = []
        weights = []
        sum_weights = 0.0

        
        # взвесить для каждой связи с сильным элементом с из С
        for j in range(row_start, row_end):
            col_idx = A.indices[j]
            if col_idx in C:
                strong_C.append(col_idx)
                weight = -A.data[j] / a_diag[f]
                weights.append(weight)
                sum_weights += weight
                
        if sum_weights > 0:
            weights = [w / sum_weights for w in weights]
            for c, w in zip(strong_C, weights):
                row.append(f)
                col.append(C_index[c])
                data.append(w)
    
    P = sparse.csr_matrix((data, (row, col)), shape=(n, n_c))
    return P
    
    

def two_level_amg(
    A: sparse.spmatrix, # input matrix
    b: np.ndarray,      # rhs vector
    x0: np.ndarray,     # initial solution approximation
    smooth_iter: int,   # count of jacobi iterations for smoothing
    theta: float,       # param for Ruge-Steben algorithm
    max_iter: int = 10000,      # maximum iteration count
    tol: float = 1e-6         # minimal residual value
):
    n = A.shape[0]
    x = x0.copy()
    
    # Определение сильных и слабых связей в матрице (вынесено за цикл!)
    C_var, F_var = ruge_stuben_variables(A, theta)
    
    # Построение интерполирующей матрицы
    P = build_interpolator(A, C_var, F_var)
    
    # Экстраполирующий оператор
    R = P.T
    
    # Определение грубой системы
    A_c = R @ A @ P
    
    for i in range(max_iter):
        # Сглаживание методом якоби для подавление высокочастотных ошибок
        x, _, _ = jacobi_solver(A, b, x, maxiter=smooth_iter)
        
        r = b - A.dot(x)
        
        # грубая невязка
        r_c = R @ r
        
        e_c = sparse.linalg.spsolve(A_c, r_c)
        
        # интерполяция с грубой сетки
        e = P @ e_c
        x = x + e
        
        # сглаживание на точной сетке
        x, _, _ = jacobi_solver(A, b, x, maxiter=smooth_iter)
        
        r_norm = np.linalg.norm(b - A.dot(x))
        
        if r_norm < tol:
            return x, i, r_norm
        
    return x, i, np.linalg.norm(b - A.dot(x))

