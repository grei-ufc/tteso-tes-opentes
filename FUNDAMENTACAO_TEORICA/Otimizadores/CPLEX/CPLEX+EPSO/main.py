import numpy as np
from model_cplex import solve_residential_dispatch
from epso import EPSO

# Dados simples da resid�ncia
load   = np.array([4, 5, 6, 4, 3, 5])
pv     = np.array([0, 1, 4, 3, 0, 0])
tariff = np.array([0.7, 0.7, 1.4, 1.4, 0.7, 0.7])

# Fun��o fitness que usa o CPLEX
def fitness(p_bat):
    # Limites de carga/descarga
    p_bat = np.clip(p_bat, -3, 3)
    return solve_residential_dispatch(load, pv, tariff, p_bat)

# Rodando o EPSO
solver = EPSO(npart=20, ngen=30, T=len(load))
best_particle, best_cost = solver.optimize(fitness)

print("\n===== RESULTADOS =====")
print("Melhor plano de bateria encontrado:", best_particle)
print("Custo m�nimo:", best_cost)
