# ============================================================
# Comparação entre PSO próprio (NumPy + Pandas) e PSO da PySwarms
# Função: Rastrigin
# Variação: número de partículas
# Autor: Douglas Barros
# ============================================================

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# Fallback para MSE sem precisar de scikit-learn
try:
    from sklearn.metrics import mean_squared_error # type: ignore
except ModuleNotFoundError:
    def mean_squared_error(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.mean((a - b) ** 2)

# Biblioteca PSO externa
import pyswarms as ps

# ------------------------------------------------------------
# Função de teste: Rastrigin
# ------------------------------------------------------------
def rastrigin(populacao: np.ndarray) -> np.ndarray:
    """Calcula a função de Rastrigin para uma população de partículas."""
    return np.sum(populacao**2 - 10 * np.cos(2 * np.pi * populacao) + 10, axis=1)


# ------------------------------------------------------------
# Implementação PSO própria (NumPy + Pandas)
# ------------------------------------------------------------
def pso_personalizado(funcao_objetivo,
                      dimensao: int,
                      minimo: float,
                      maximo: float,
                      n_particulas: int = 4000,
                      max_epocas: int = 4000,
                      w: float = 0.729,
                      c1: float = 1.49445,
                      c2: float = 1.49445,
                      vmax_frac: float = 0.2,
                      semente: int | None = None):
    """Implementação moderna do PSO utilizando NumPy e Pandas."""
    rng = np.random.default_rng(semente)

    posicoes = rng.uniform(minimo, maximo, size=(n_particulas, dimensao))
    vmax = (maximo - minimo) * vmax_frac
    velocidades = rng.uniform(-vmax, vmax, size=(n_particulas, dimensao))

    valores = funcao_objetivo(posicoes)
    melhor_pos_pessoal = posicoes.copy()
    melhor_valor_pessoal = valores.copy()

    idx_melhor_global = np.argmin(melhor_valor_pessoal)
    melhor_pos_global = melhor_pos_pessoal[idx_melhor_global].copy()
    melhor_valor_global = melhor_valor_pessoal[idx_melhor_global]

    historico = []

    for epoca in range(max_epocas):
        r1 = rng.random((n_particulas, dimensao))
        r2 = rng.random((n_particulas, dimensao))

        velocidades = (w * velocidades
                       + c1 * r1 * (melhor_pos_pessoal - posicoes)
                       + c2 * r2 * (melhor_pos_global - posicoes))

        velocidades = np.clip(velocidades, -vmax, vmax)
        posicoes += velocidades
        posicoes = np.clip(posicoes, minimo, maximo)

        valores = funcao_objetivo(posicoes)

        melhora = valores < melhor_valor_pessoal
        melhor_pos_pessoal[melhora] = posicoes[melhora]
        melhor_valor_pessoal[melhora] = valores[melhora]

        idx = np.argmin(melhor_valor_pessoal)
        if melhor_valor_pessoal[idx] < melhor_valor_global:
            melhor_valor_global = melhor_valor_pessoal[idx]
            melhor_pos_global = melhor_pos_pessoal[idx].copy()

        historico.append(melhor_valor_global)

    return melhor_pos_global, melhor_valor_global, historico


# ------------------------------------------------------------
# PSO da biblioteca PySwarms
# ------------------------------------------------------------
def pso_pyswarms(dimensao: int,
                 n_particulas: int = 4000,
                 max_epocas: int = 4000,
                 minimo: float = -5.12,
                 maximo: float = 5.12):
    """Executa o PSO da biblioteca PySwarms com a função Rastrigin."""
    from pyswarms.single import GlobalBestPSO
    options = {'c1': 1.49445, 'c2': 1.49445, 'w': 0.729}

    optimizer = GlobalBestPSO(
        n_particles=n_particulas,
        dimensions=dimensao,
        options=options,
        bounds=(minimo * np.ones(dimensao), maximo * np.ones(dimensao))
    )

    melhor_custo, melhor_pos = optimizer.optimize(rastrigin, iters=max_epocas, verbose=False)

    return melhor_pos, melhor_custo, optimizer.cost_history


# ------------------------------------------------------------
# Função de comparação para vários tamanhos de população
# ------------------------------------------------------------
def comparar_pso(dimensao=20, epocas=4000, particulas_lista=None):
    if particulas_lista is None:
        particulas_lista = [100, 500, 1000, 2000, 4000]

    resultados = []

    for n_particulas in particulas_lista:
        print(f"\n--- Teste com {n_particulas} partículas ---")

        # PSO personalizado
        inicio = time.time()
        _, valor_np, hist_np = pso_personalizado(
            rastrigin, dimensao, -5.12, 5.12,
            n_particulas=n_particulas, max_epocas=epocas, semente=42
        )
        tempo_np = time.time() - inicio

        # PSO PySwarms
        inicio = time.time()
        _, valor_ps, hist_ps = pso_pyswarms(
            dimensao, n_particulas=n_particulas, max_epocas=epocas
        )
        tempo_ps = time.time() - inicio

        # Igualar tamanhos
        min_len = min(len(hist_np), len(hist_ps))
        h1 = np.array(hist_np[:min_len])
        h2 = np.array(hist_ps[:min_len])

        # Erro médio quadrático
        mse = mean_squared_error(h1, h2)

        resultados.append({
            "Partículas": n_particulas,
            "Melhor valor (NumPy+Pandas)": valor_np,
            "Melhor valor (PySwarms)": valor_ps,
            "Tempo NumPy+Pandas (s)": tempo_np,
            "Tempo PySwarms (s)": tempo_ps,
            "MSE": mse
        })

        print(f"PSO Personalizado → valor = {valor_np:.6f} | tempo = {tempo_np:.2f}s")
        print(f"PSO PySwarms      → valor = {valor_ps:.6f} | tempo = {tempo_ps:.2f}s")
        print(f"Erro médio quadrático (MSE): {mse:.6e}")

    df_resultados = pd.DataFrame(resultados)
    print("\n=== RESUMO FINAL ===")
    print(df_resultados)

    # Gráficos comparativos
    plt.figure(figsize=(10, 6))
    plt.plot(df_resultados["Partículas"], df_resultados["MSE"], 'o-', label="Erro Médio Quadrático (MSE)")
    plt.xlabel("Número de Partículas")
    plt.ylabel("Erro Médio Quadrático")
    plt.title("Comparação PSO NumPy+Pandas vs PySwarms (Erro Médio Quadrático)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(df_resultados["Partículas"], df_resultados["Tempo NumPy+Pandas (s)"], 'o-', label="Tempo NumPy+Pandas")
    plt.plot(df_resultados["Partículas"], df_resultados["Tempo PySwarms (s)"], 'o-', label="Tempo PySwarms")
    plt.xlabel("Número de Partículas")
    plt.ylabel("Tempo de Execução (s)")
    plt.title("Tempo de Execução por Tamanho da População")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df_resultados


# ------------------------------------------------------------
# Execução principal
# ------------------------------------------------------------
if __name__ == "__main__":
    resultados = comparar_pso(dimensao=50, epocas=5000,
                              particulas_lista=[100, 500, 1000, 2500, 5000])
