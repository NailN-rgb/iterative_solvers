import numpy as np
from scipy import sparse
from time import perf_counter


from solvers.amg import two_level_amg

from problems.cubic_spline import cubic_spline


# Basic test. Generate matrix of fixed size & check converganse and error with correct solution 

n = 1000
smooth_iters = 3
theta = 0.5
repeats = 10

A, b = cubic_spline(n + 1)
x0 = np.random.rand(A.shape[0])

# Точное решение
correct_sol = sparse.linalg.spsolve(A, b)


def measure_time(method_name, solver_fn, *args, **kwargs):
    total_time = 0.0
    total_error_norm = 0.0 
    for _ in range(repeats):
        start = perf_counter()
        sol, _, _ = solver_fn(*args, **kwargs)
        end = perf_counter()
        total_time += (end - start)
        diff_norm =+ np.linalg.norm(sol - correct_sol)
    
    avg_time = total_time / repeats
    total_error_norm = diff_norm / repeats
    print(f"[{method_name:<10}] Среднее время: {avg_time:.6f} сек, средняя ошибка {total_error_norm:.16f}")
    
    
measure_time("AMG", two_level_amg, A, b, x0, smooth_iters, theta)

