# epso_unico.py
# EPSO corrigido para minimizar Rastrigin
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

def rastrigin_population(pop):
    A = 10.0
    return A * pop.shape[1] + np.sum(pop**2 - A * np.cos(2 * np.pi * pop), axis=1)

def epso(funcao_objetivo, npar=500, nvar=7, T=0.1, max_it=500,
         minimo=-5.12, maximo=5.12, weight_clip=5.0, vmax_frac=0.2,
         semente=None, verbose=False):
    rng = np.random.default_rng(semente)
    Particulas = rng.uniform(minimo, maximo, (npar, nvar))
    vmax = (maximo - minimo) * vmax_frac
    Velocities = rng.uniform(-vmax, vmax, (npar, nvar))

    Weighta = np.ones((npar, nvar)) * 0.9
    Weightb = np.ones((npar, nvar)) * 1.0
    Weightc = np.ones((npar, nvar)) * 1.0

    Personal_bests = Particulas.copy()
    Fitness_Personal_Bests = funcao_objetivo(Particulas)
    idx = int(np.argmin(Fitness_Personal_Bests))
    Fitness_global_best = float(Fitness_Personal_Bests[idx])
    global_best = Personal_bests[idx].copy()

    history = []

    start = time.time()
    w_schedule = 2 - ((2.2 - 0.4) / max_it) * (np.arange(max_it) / max_it)
    for it in range(max_it):
        Noisea = rng.normal(0, 1, size=(npar, nvar)) * (T * 0.1)
        Noiseb = rng.normal(0, 1, size=(npar, nvar)) * (T * 0.1)
        Noisec = rng.normal(0, 1, size=(npar, nvar)) * (T * 0.1)

        Weighta = Weighta + Noisea * w_schedule[it]
        Weightb = Weightb + Noiseb * 0.5 * T
        Weightc = Weightc + Noisec * 0.5 * T

        Weighta = np.clip(Weighta, -weight_clip, weight_clip)
        Weightb = np.clip(Weightb, -weight_clip, weight_clip)
        Weightc = np.clip(Weightc, -weight_clip, weight_clip)

        a = Weighta * Velocities
        b = Weightb * (Personal_bests - Particulas)
        c = Weightc * (global_best - Particulas)

        Velocities = a + b + c
        Velocities = np.clip(Velocities, -vmax, vmax)

        Particulas = np.clip(Particulas + Velocities, minimo, maximo)

        Eval = funcao_objetivo(Particulas)
        improved_mask = Eval < Fitness_Personal_Bests
        if np.any(improved_mask):
            Fitness_Personal_Bests[improved_mask] = Eval[improved_mask]
            Personal_bests[improved_mask] = Particulas[improved_mask].copy()

        min_index = int(np.argmin(Fitness_Personal_Bests))
        if Fitness_Personal_Bests[min_index] < Fitness_global_best:
            Fitness_global_best = float(Fitness_Personal_Bests[min_index])
            global_best = Personal_bests[min_index].copy()

        history.append(float(Fitness_global_best))
        if verbose and it % 50 == 0:
            print(f'[EPSO] Iter {it:4d} best = {Fitness_global_best:.6f}')

    elapsed = time.time() - start
    return global_best, float(Fitness_global_best), np.array(history), elapsed

if __name__ == "__main__":
    best_pos, best_val, hist, t = epso(rastrigin_population, verbose=True)
    print("\\nEPSO finished in %.2f s — best = %.6f" % (t, best_val))
    plt.plot(hist)
    plt.xlabel("Iterações")
    plt.ylabel("Melhor Fitness Global")
    plt.title("EPSO - Rastrigin")
    plt.grid(True)
    plt.show()
