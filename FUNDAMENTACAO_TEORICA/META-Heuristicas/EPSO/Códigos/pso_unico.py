# pso_unico.py
# PSO clássico para minimizar Rastrigin
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

def rastrigin_population(pop):
    A = 10.0
    return A * pop.shape[1] + np.sum(pop**2 - A * np.cos(2 * np.pi * pop), axis=1)

def pso(funcao_objetivo, dimensao=7, minimo=-5.12, maximo=5.12,
        n_particulas=500, max_epocas=500, w=0.729, c1=1.49445, c2=1.49445,
        vmax_frac=0.2, semente=None, verbose=False):
    rng = np.random.default_rng(semente)
    posicoes = rng.uniform(minimo, maximo, (n_particulas, dimensao))
    vmax = (maximo - minimo) * vmax_frac
    velocidades = rng.uniform(-vmax, vmax, (n_particulas, dimensao))

    valores = funcao_objetivo(posicoes)
    melhor_pessoal = posicoes.copy()
    melhor_valor_pessoal = valores.copy()

    idx_gbest = np.argmin(melhor_valor_pessoal)
    melhor_global = melhor_pessoal[idx_gbest].copy()
    melhor_valor_global = melhor_valor_pessoal[idx_gbest]

    historico = []

    start = time.time()
    for epoca in range(max_epocas):
        r1 = rng.random((n_particulas, dimensao))
        r2 = rng.random((n_particulas, dimensao))

        velocidades = (
            w * velocidades
            + c1 * r1 * (melhor_pessoal - posicoes)
            + c2 * r2 * (melhor_global - posicoes)
        )
        velocidades = np.clip(velocidades, -vmax, vmax)
        posicoes = np.clip(posicoes + velocidades, minimo, maximo)

        valores = funcao_objetivo(posicoes)
        melhora = valores < melhor_valor_pessoal
        if np.any(melhora):
            melhor_pessoal[melhora] = posicoes[melhora].copy()
            melhor_valor_pessoal[melhora] = valores[melhora]

        idx_gbest = np.argmin(melhor_valor_pessoal)
        if melhor_valor_pessoal[idx_gbest] < melhor_valor_global:
            melhor_global = melhor_pessoal[idx_gbest].copy()
            melhor_valor_global = melhor_valor_pessoal[idx_gbest]

        historico.append(float(melhor_valor_global))
        if verbose and epoca % 50 == 0:
            print(f'[PSO] Iter {epoca:4d} best = {melhor_valor_global:.6f}')

    elapsed = time.time() - start
    return melhor_global, float(melhor_valor_global), np.array(historico), elapsed

if __name__ == "__main__":
    best_pos, best_val, hist, t = pso(rastrigin_population, verbose=True)
    print("\\nPSO finished in %.2f s — best = %.6f" % (t, best_val))
    plt.plot(hist)
    plt.xlabel("Iterações")
    plt.ylabel("Melhor Fitness Global")
    plt.title("PSO - Rastrigin")
    plt.grid(True)
    plt.show()
