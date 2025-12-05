# =============================================================
# comparacao_pso_epso.py
# =============================================================
# Comparação experimental entre os algoritmos PSO e EPSO
# Ambos otimizam a função de Rastrigin com os mesmos parâmetros.
# O código executa múltiplos testes (runs), coleta os resultados,
# calcula as curvas médias de convergência e compara desempenho.
#
# Também é calculado o erro quadrático médio (MSE) entre as curvas
# médias de convergência do PSO e do EPSO, para medir diferença
# no comportamento de convergência.
#
# Autor: Douglas Barros + ChatGPT (GPT-5)
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import os

# Importa as funções dos arquivos individuais
from pso_unico import pso, rastrigin_population
from epso_unico import epso

# -------------------------------------------------------------
# PARÂMETROS DO EXPERIMENTO
# -------------------------------------------------------------
DIM = 7           # Número de dimensões (variáveis da função)
N_PARTICLES = 500   # Número de partículas
N_ITERS = 500       # Número de iterações (épocas)
N_RUNS = 30         # Número de execuções independentes

# Diretório para salvar resultados e gráficos
out_dir = "projeto_epso_pso_results"
os.makedirs(out_dir, exist_ok=True)

# Listas para armazenar resultados
resultados = []          # Guarda os valores finais e tempos de cada run
curvas_pso = []          # Guarda a curva de convergência do PSO por run
curvas_epso = []         # Guarda a curva de convergência do EPSO por run

# -------------------------------------------------------------
# EXECUÇÃO DOS EXPERIMENTOS
# -------------------------------------------------------------
inicio_geral = time.time()

for run in range(N_RUNS):
    semente = 1000 + run

    # --- Execução do PSO ---
    pso_best_pos, pso_best_val, pso_hist, pso_time = pso(
        rastrigin_population,
        dimensao=DIM,
        n_particulas=N_PARTICLES,
        max_epocas=N_ITERS,
        semente=semente,
        verbose=False
    )

    # --- Execução do EPSO ---
    epso_best_pos, epso_best_val, epso_hist, epso_time = epso(
        rastrigin_population,
        npar=N_PARTICLES,
        nvar=DIM,
        max_it=N_ITERS,
        semente=semente + 1000,
        verbose=False
    )

    # --- Exibição do progresso ---
    print(f"Execução {run+1:02d}/{N_RUNS} → "
          f"PSO = {pso_best_val:.6f} | EPSO = {epso_best_val:.6f}")

    # Armazena resultados finais e históricos
    resultados.append((run+1, pso_best_val, epso_best_val, pso_time, epso_time))
    curvas_pso.append(pso_hist)
    curvas_epso.append(epso_hist)

# Tempo total do experimento
tempo_total = time.time() - inicio_geral

# -------------------------------------------------------------
# SALVAMENTO DOS RESULTADOS EM CSV
# -------------------------------------------------------------
csv_path = os.path.join(out_dir, "comparacao_resultados.csv")
with open(csv_path, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["execucao", "melhor_PSO", "melhor_EPSO", "tempo_PSO_s", "tempo_EPSO_s"])
    for linha in resultados:
        writer.writerow(linha)

# -------------------------------------------------------------
# CÁLCULO DAS CURVAS MÉDIAS DE CONVERGÊNCIA
# -------------------------------------------------------------
media_pso = np.mean(np.vstack(curvas_pso), axis=0)
media_epso = np.mean(np.vstack(curvas_epso), axis=0)

# -------------------------------------------------------------
# MÉTRICA DE DESEMPENHO: ERRO QUADRÁTICO MÉDIO (MSE)
# -------------------------------------------------------------
mse = np.mean((media_pso - media_epso)**2)

# -------------------------------------------------------------
# GRÁFICO DE CONVERGÊNCIA MÉDIA
# -------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(media_pso, label="PSO (média de 30 execuções)", linewidth=2)
plt.plot(media_epso, label="EPSO (média de 30 execuções)", linewidth=2, linestyle="--")
plt.xlabel("Iterações", fontsize=12)
plt.ylabel("Melhor valor da função (fitness)", fontsize=12)
plt.title(f"Comparação PSO vs EPSO — Função Rastrigin ({DIM}D)\nMSE = {mse:.6e}", fontsize=13)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "convergencia_media.png"), dpi=300)
plt.show()

# -------------------------------------------------------------
# BOXPLOT DOS RESULTADOS FINAIS
# -------------------------------------------------------------
valores_pso = [r[1] for r in resultados]
valores_epso = [r[2] for r in resultados]

plt.figure(figsize=(8, 6))
plt.boxplot([valores_pso, valores_epso],
            labels=["PSO", "EPSO"],
            patch_artist=True,
            boxprops=dict(facecolor="lightblue"),
            medianprops=dict(color="red", linewidth=2))
plt.ylabel("Melhor valor final (função Rastrigin)", fontsize=12)
plt.title("Distribuição dos resultados finais — 30 execuções", fontsize=13)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "boxplot_resultados_finais.png"), dpi=300)
plt.show()

# -------------------------------------------------------------
# RESUMO FINAL
# -------------------------------------------------------------
media_pso_final = np.mean(valores_pso)
media_epso_final = np.mean(valores_epso)
desvio_pso = np.std(valores_pso)
desvio_epso = np.std(valores_epso)

print("\n========== RESUMO DO EXPERIMENTO ==========")
print(f"Dimensões: {DIM} | Partículas: {N_PARTICLES} | Iterações: {N_ITERS}")
print(f"Número de execuções: {N_RUNS}")
print(f"Tempo total de execução: {tempo_total:.2f} segundos\n")
print(f"PSO → Média: {media_pso_final:.6f} | Desvio padrão: {desvio_pso:.6f}")
print(f"EPSO → Média: {media_epso_final:.6f} | Desvio padrão: {desvio_epso:.6f}")
print(f"\nErro Quadrático Médio (MSE) entre curvas médias: {mse:.6e}")
print(f"Resultados e gráficos salvos em: '{out_dir}'")
print("===========================================")
