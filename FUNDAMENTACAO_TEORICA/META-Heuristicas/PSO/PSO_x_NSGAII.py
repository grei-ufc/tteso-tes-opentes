# --------------------------------------------------------
# **Comparação PSO vs NSGA-II no problema ZDT1**
# --------------------------------------------------------

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems import get_problem
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 1. Definir o problema multiobjetivo (ZDT1)
# --------------------------------------------------------
problem = get_problem("zdt1")  # n_var = 30, f1(x), f2(x)

# --------------------------------------------------------
# 2. Rodar o NSGA-II
# --------------------------------------------------------
algorithm_nsga2 = NSGA2(pop_size=100)

res_nsga2 = minimize(
    problem,
    algorithm_nsga2,
    ('n_gen', 200),
    seed=1,
    verbose=False
)

# --------------------------------------------------------
# 3. Implementar PSO multiobjetivo (via soma ponderada)
# --------------------------------------------------------

def pso_multiobjective(problem, num_particles=50, num_iterations=200, w=0.7, c1=1.5, c2=1.5):
    n_var = problem.n_var
    lb, ub = problem.xl, problem.xu

    # Inicialização
    X = np.random.rand(num_particles, n_var) * (ub - lb) + lb
    V = np.zeros_like(X)
    personal_best = X.copy()
    personal_best_val = np.full((num_particles,), np.inf)
    global_best = None
    global_best_val = np.inf

    # Pesos aleatórios entre os objetivos (cada partícula tem preferências distintas)
    weights = np.random.dirichlet(np.ones(problem.n_obj), size=num_particles)

    for it in range(num_iterations):
        # Avaliar partículas (retorna vetor [f1, f2])
        F = np.array([problem.evaluate(x) for x in X])

        # Converter F multiobjetivo em escalar via soma ponderada
        scalar_fitness = np.array([
            np.dot(weights[i], F[i]) for i in range(num_particles)
        ])

        # Atualizar melhores pessoais
        for i in range(num_particles):
            if scalar_fitness[i] < personal_best_val[i]:
                personal_best[i] = X[i].copy()
                personal_best_val[i] = scalar_fitness[i]

        # Atualizar melhor global
        min_idx = np.argmin(scalar_fitness)
        if scalar_fitness[min_idx] < global_best_val:
            global_best_val = scalar_fitness[min_idx]
            global_best = X[min_idx].copy()

        # Atualizar velocidades e posições
        r1, r2 = np.random.rand(num_particles, n_var), np.random.rand(num_particles, n_var)
        V = w * V + c1 * r1 * (personal_best - X) + c2 * r2 * (global_best - X)
        X = np.clip(X + V, lb, ub)

    # Avaliar soluções finais
    F_final = np.array([problem.evaluate(x) for x in X])
    return X, F_final

# Executar PSO multiobjetivo
X_pso, F_pso = pso_multiobjective(problem)

# --------------------------------------------------------
# 4. Fronteira teórica de Pareto (f2 = 1 - sqrt(f1))
# --------------------------------------------------------
f1_teorico = np.linspace(0, 1, 200)
f2_teorico = 1 - np.sqrt(f1_teorico)

# --------------------------------------------------------
# 5. Visualização dos resultados
# --------------------------------------------------------
F_nsga2 = res_nsga2.F

plt.figure(figsize=(9, 6))
plt.scatter(F_nsga2[:, 0], F_nsga2[:, 1], s=20, c='blue', label='NSGA-II (fronteira aproximada)')
plt.scatter(F_pso[:, 0], F_pso[:, 1], s=20, c='red', alpha=0.7, label='PSO (soma ponderada)')
plt.plot(f1_teorico, f2_teorico, 'k--', linewidth=2, label='Fronteira teórica de Pareto')

plt.xlabel("f₁(x): minimizar o primeiro objetivo")
plt.ylabel("f₂(x): minimizar o segundo objetivo")
plt.title("Comparação NSGA-II vs PSO no problema ZDT1")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
