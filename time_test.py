import numpy as np
from scipy import sparse
from time import perf_counter

from solvers.richardson import simple_iteration_solver
from solvers.jacobi import jacobi_solver
from solvers.sor import sor_solver
from solvers.ssor import ssor_solver
from solvers.cg import c_gradient
from solvers.pcg import pcg
from solvers.amg import two_level_amg

from problems.cubic_spline import cubic_spline

n = 1000  # размерность СЛАУ
repeats = 10  # число повторений

A, b = cubic_spline(n)
x0 = np.random.rand(A.shape[0])

def measure_time(method_name, solver_fn, *args, **kwargs):
    total_time = 0.0
    for _ in range(repeats):
        start = perf_counter()
        solver_fn(*args, **kwargs)
        end = perf_counter()
        total_time += (end - start)
    
    avg_time = total_time / repeats
    print(f"[{method_name:<10}] Среднее время: {avg_time:.6f} сек")

# ------------------- Замер времени -------------------

print(f"\nИзмерение времени работы на системе размерности n = {n}, повторов: {repeats}\n")

measure_time("Richardson", simple_iteration_solver, A, b, x0)
measure_time("Jacobi",     jacobi_solver,          A, b, x0)
measure_time("SOR",        sor_solver,             A, b, x0)
measure_time("SSOR",       ssor_solver,            A, b, x0)
measure_time("CG",         c_gradient,             A, b, x0)
measure_time("PCG",        pcg,                    A, b, x0)
measure_time("AMG",        two_level_amg,          A, b, x0, 2, 0.25)

print("\nТестирование завершено.")
