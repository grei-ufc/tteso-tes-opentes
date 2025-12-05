# ============================================================
# EPSO (Evolutionary Particle Swarm Optimization)
# Minimização da Função de Rastrigin
# Autor: Douglas Barros
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Função de avaliação — Rastrigin
# ------------------------------------------------------------
def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# ------------------------------------------------------------
# EPSO (versão corrigida e estável)
# ------------------------------------------------------------
def epso(
    funcao_objetivo,
    npar=500,
    nvar=7,
    T=0.5,          # taxa de mutação
    max_it=500,
    minimo=-5.12,
    maximo=5.12,
    weight_clip=(0.5, 2.5),
    vmax_frac=0.2,
    semente=None,
    verbose=False
):
    rng = np.random.default_rng(semente)

    particula = rng.uniform(minimo, maximo, nvar)
    global_best = particula.copy()
    Particulas = rng.uniform(minimo, maximo, (npar, nvar))
    Velocities = rng.uniform(-(maximo - minimo) * vmax_frac, (maximo - minimo) * vmax_frac, (npar, nvar))

    Weighta = rng.uniform(0.8, 1.2, (npar, nvar))
    Weightb = rng.uniform(0.8, 1.2, (npar, nvar))
    Weightc = rng.uniform(1.5, 2.5, (npar, nvar))

    Personal_bests = Particulas.copy()
    Fitness_Personal_Bests = np.array([funcao_objetivo(p) for p in Particulas])
    Fitness_global_best = np.min(Fitness_Personal_Bests)
    global_best = Particulas[np.argmin(Fitness_Personal_Bests)].copy()

    best_cost_iteration = np.zeros(max_it)
    w = 2 - ((2.2 - 0.4) / max_it) * (np.arange(max_it) / max_it)

    for it in range(max_it):
        Na = rng.normal(0, 1, (npar, nvar))
        Nb = rng.normal(0, 1, (npar, nvar))
        Nc = rng.normal(0, 1, (npar, nvar))

        Weighta = np.clip(Weighta + Na * T, *weight_clip)
        Weightb = np.clip(Weightb + Nb * T, *weight_clip)
        Weightc = np.clip(Weightc + Nc * T, *weight_clip)

        a = Weighta * Velocities
        b = Weightb * (Personal_bests - Particulas)
        c = Weightc * (global_best - Particulas)

        Velocities = a + b + c
        vmax = (maximo - minimo) * vmax_frac
        Velocities = np.clip(Velocities, -vmax, vmax)

        Particulas = np.clip(Particulas + Velocities, minimo, maximo)

        Eval = np.array([funcao_objetivo(p) for p in Particulas])

        improved = Eval < Fitness_Personal_Bests
        Fitness_Personal_Bests[improved] = Eval[improved]
        Personal_bests[improved] = Particulas[improved].copy()

        min_index = np.argmin(Fitness_Personal_Bests)
        if Fitness_Personal_Bests[min_index] < Fitness_global_best:
            Fitness_global_best = Fitness_Personal_Bests[min_index]
            global_best = Personal_bests[min_index].copy()

        particula = global_best.copy()
        best_cost_iteration[it] = Fitness_global_best

        if verbose and it % 10 == 0:
            print(f"Iteração {it:3d} | Melhor valor = {Fitness_global_best:.6f}")

    return global_best, Fitness_global_best, best_cost_iteration


# ------------------------------------------------------------
# Execução
# ------------------------------------------------------------
if __name__ == "__main__":
    best_pos, best_val, hist = epso(rastrigin, verbose=True)
    print("\nMelhor valor encontrado:", best_val)
    plt.plot(hist)
    plt.xlabel("Iterações")
    plt.ylabel("Melhor Fitness Global")
    plt.title("Convergência EPSO - Função Rastrigin")
    plt.grid(True)
    plt.show()
