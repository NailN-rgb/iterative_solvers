import numpy as np
from scipy import sparse

from solvers.richardson import simple_iteration_solver
from solvers.jacobi import jacobi_solver
from solvers.sor import sor_solver
from solvers.ssor import ssor_solver
from solvers.cg import c_gradient
from solvers.amg import two_level_amg

from problems.FEMSystem import laplase_problem_generator


# Test on system for Laplase equation 

n = 200
# A, b = generate_spd_matrix(n)
# b = np.random.rand(n)
A, b = laplase_problem_generator(n + 1)
x0 = np.random.rand(A.shape[0])

# Точное решение
correct_sol = sparse.linalg.spsolve(A, b)


def print_result(name, sol, iters, res, correct_sol):
    diff_norm = np.linalg.norm(sol - correct_sol)
    if iters == -1:
        print(f"[{name:<10}] Не сошлось за максимальное число итераций. ||x - x*|| = {diff_norm:.2e}, остаток = {res:.2e}")
    else:
        print(f"[{name:<10}] Итераций: {iters:4d}, ||x - x*|| = {diff_norm:.2e}, остаток = {res:.2e}")

# Turned off for laplace problem
# print_result("Richardson", *simple_iteration_solver(A, b, x0), correct_sol)

print_result("Jacobi",     *jacobi_solver(A, b, x0), correct_sol)
print_result("SOR",        *sor_solver(A, b, x0), correct_sol)
print_result("SSOR",       *ssor_solver(A, b, x0), correct_sol)
print_result("CG",         *c_gradient(A, b, x0), correct_sol)
print_result("AMG",        *two_level_amg(A, b, x0, 2, 0.25), correct_sol)

print("\nПрограмма завершена.")
