import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from solvers.jacobi import jacobi_solver
from solvers.richardson import simple_iteration_solver
from solvers.sor import sor_solver
from solvers.ssor import ssor_solver
from solvers.amg import two_level_amg

from problems.cubic_spline import cubic_spline

# Plot abs error values for each solution menthod

def track_convergence(solver, A, b, x0, correct_sol, max_total_iters=1000, tol=1e-6, **kwargs):
    x = x0.copy()
    errors = []

    for _ in range(max_total_iters):
        error = np.linalg.norm(x - correct_sol)
        errors.append(error)

        # Одношаговый вызов метода
        x_new, _, res = solver(A, b, x, tol=0, maxiter=1, **kwargs)

        if np.linalg.norm(A @ x_new - b) < tol:
            errors.append(np.linalg.norm(x_new - correct_sol))
            break

        x = x_new

    return errors

def track_convergence_amg(solver, A, b, x0, correct_sol, smooth = 2, theta = 0.25, max_total_iters=1000, tol=1e-6, **kwargs):
    x = x0.copy()
    errors = []

    for _ in range(max_total_iters):
        error = np.linalg.norm(x - correct_sol)
        errors.append(error)

        # Одношаговый вызов метода
        x_new, _, res = solver(A, b, x, smooth, theta, tol=0, maxiter=1, **kwargs)

        if np.linalg.norm(A @ x_new - b) < tol:
            errors.append(np.linalg.norm(x_new - correct_sol))
            break

        x = x_new

    return errors

#---------------------------------------------------------------------------------------

n = 1000
A, b = cubic_spline(n + 1)
x0 = np.zeros(n)
correct_sol = sparse.linalg.spsolve(A, b)

# --- Трекинг ---
jacobi_errors     = track_convergence(jacobi_solver, A, b, x0, correct_sol)
richardson_errors = track_convergence(simple_iteration_solver, A, b, x0, correct_sol)
sor_errors        = track_convergence(sor_solver, A, b, x0, correct_sol)
ssor_errors       = track_convergence(ssor_solver, A, b, x0, correct_sol)
amg_errors        = track_convergence_amg(two_level_amg, A, b, x0, correct_sol)

# --- Визуализация ---
plt.figure(figsize=(10, 6))
plt.plot(jacobi_errors, label="Jacobi")
plt.plot(richardson_errors, label="Richardson")
plt.plot(sor_errors, label="SOR")
plt.plot(ssor_errors, label="SSOR")
plt.plot(amg_errors, label="AMG")
plt.yscale("log")
plt.xlabel("Итерация")
plt.ylabel("||x_k - x*||")
plt.title("Отклонение от точного решения")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/convergence track.png")