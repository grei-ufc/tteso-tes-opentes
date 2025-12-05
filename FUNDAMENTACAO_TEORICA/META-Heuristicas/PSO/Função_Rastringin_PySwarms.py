# ============================================================
# Comparação entre PSO próprio (NumPy + Pandas) e PSO da PySwarms
# Função: Rastrigin
# Autor: Douglas Barros
# ============================================================

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# Biblioteca PSO externa
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history

# ------------------------------------------------------------
# Função de teste: Rastrigin
# ------------------------------------------------------------
def rastrigin(populacao: np.ndarray) -> np.ndarray:
    """
    Calcula a função de Rastrigin para uma população de partículas.
    populacao: array (n_particulas, dimensao)
    retorna: array (n_particulas,) com o valor de cada partícula
    """
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
                      semente: int | None = None,
                      verbose: bool = True):
    """
    Implementação moderna do PSO utilizando NumPy e Pandas.
    """
    rng = np.random.default_rng(semente)

    # Inicialização
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

    # Loop principal
    for epoca in range(max_epocas):
        r1 = rng.random((n_particulas, dimensao))
        r2 = rng.random((n_particulas, dimensao))

        # Atualização de velocidades
        velocidades = (w * velocidades
                       + c1 * r1 * (melhor_pos_pessoal - posicoes)
                       + c2 * r2 * (melhor_pos_global - posicoes))

        # Limite de velocidade
        velocidades = np.clip(velocidades, -vmax, vmax)

        # Atualização de posições
        posicoes += velocidades
        posicoes = np.clip(posicoes, minimo, maximo)

        # Avaliar nova posição
        valores = funcao_objetivo(posicoes)

        # Atualização de melhores pessoais
        melhora = valores < melhor_valor_pessoal
        melhor_pos_pessoal[melhora] = posicoes[melhora]
        melhor_valor_pessoal[melhora] = valores[melhora]

        # Atualização de melhor global
        idx = np.argmin(melhor_valor_pessoal)
        if melhor_valor_pessoal[idx] < melhor_valor_global:
            melhor_valor_global = melhor_valor_pessoal[idx]
            melhor_pos_global = melhor_pos_pessoal[idx].copy()

        # Guardar histórico
        historico.append({
            "época": epoca,
            "melhor_valor_global": melhor_valor_global
        })

        if verbose and epoca % 10 == 0:
            print(f"Época {epoca:4d} | Melhor valor global = {melhor_valor_global:.6f}")

    df_historico = pd.DataFrame(historico)
    return melhor_pos_global, melhor_valor_global, df_historico


# ------------------------------------------------------------
# PSO da biblioteca PySwarms
# ------------------------------------------------------------
def pso_pyswarms(dimensao: int,
                 n_particulas: int = 4000,
                 max_epocas: int = 4000,
                 minimo: float = -5.12,
                 maximo: float = 5.12):
    """
    Executa o PSO da biblioteca PySwarms com a função Rastrigin.
    """
    from pyswarms.single import GlobalBestPSO

    options = {'c1': 1.49445, 'c2': 1.49445, 'w': 0.729}
    optimizer = GlobalBestPSO(
        n_particles=n_particulas,
        dimensions=dimensao,
        options=options,
        bounds=(minimo * np.ones(dimensao), maximo * np.ones(dimensao))
    )

    # Otimização
    melhor_custo, melhor_pos = optimizer.optimize(rastrigin, iters=max_epocas, verbose=False)

    return melhor_pos, melhor_custo, optimizer.cost_history


# ------------------------------------------------------------
# Execução e comparação
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== COMPARAÇÃO ENTRE PSO PERSONALIZADO E PSO PySwarms ===\n")

    dimensao = 20
    n_particulas = 4000
    max_epocas = 4000

    print(f"Função: Rastrigin ({dimensao} dimensões)")
    print(f"Configuração: {n_particulas} partículas, {max_epocas} épocas\n")

    # -------------------------
    # 1️⃣ PSO personalizado
    # -------------------------
    print("Rodando PSO personalizado (NumPy + Pandas)...")
    inicio = time.time()
    melhor_pos1, melhor_valor1, hist1 = pso_personalizado(
        rastrigin,
        dimensao,
        minimo=-5.12,
        maximo=5.12,
        n_particulas=n_particulas,
        max_epocas=max_epocas,
        semente=42,
        verbose=False
    )
    tempo1 = time.time() - inicio
    print(f"→ Concluído em {tempo1:.3f}s | Melhor valor = {melhor_valor1:.6f}\n")

    # -------------------------
    # 2️⃣ PSO da PySwarms
    # -------------------------
    print("Rodando PSO da biblioteca PySwarms...")
    inicio = time.time()
    melhor_pos2, melhor_valor2, hist2 = pso_pyswarms(
        dimensao,
        n_particulas=n_particulas,
        max_epocas=max_epocas
    )
    tempo2 = time.time() - inicio
    print(f"→ Concluído em {tempo2:.3f}s | Melhor valor = {melhor_valor2:.6f}\n")

    # -------------------------
    # 3️⃣ Comparação resumida
    # -------------------------
    print("=== RESULTADOS FINAIS ===")
    print(f"PSO Personalizado  → Erro final: {melhor_valor1:.6f} | Tempo: {tempo1:.3f} s")
    print(f"PSO PySwarms       → Erro final: {melhor_valor2:.6f} | Tempo: {tempo2:.3f} s")

    # -------------------------
    # 4️⃣ Gráficos comparativos
    # -------------------------
    plt.figure(figsize=(10,5))
    plt.plot(hist1["época"], hist1["melhor_valor_global"], label="PSO (NumPy + Pandas)")
    plt.plot(np.arange(len(hist2)), hist2, label="PSO (PySwarms)")
    plt.xlabel("Iterações")
    plt.ylabel("Melhor valor encontrado")
    plt.title(f"Convergência - Função Rastrigin ({dimensao} dimensões)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
