# ============================================================
# EPSO para avaliação via Função de Rastrigin
# Adaptação do código original de alocação de baterias
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Parâmetros principais do EPSO
# -------------------------------
npar = 10            # Número de partículas na população
nvar = 7             # Dimensão da partícula (número de variáveis)
T = 1.0              # Taxa de mutação
max_it = 500         # Número máximo de iterações

# -------------------------------
# Função objetivo (Rastrigin)
# -------------------------------
def rastrigin(x):
    """Função Rastrigin (mínimo global em x=0, f(x)=0)."""
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# -------------------------------
# Funções auxiliares
# -------------------------------
def round_half_up(x):
    """Arredonda corretamente números e arrays."""
    if isinstance(x, (int, float)):
        return int(np.floor(x + 0.5))
    elif isinstance(x, np.ndarray):
        return np.floor(x + 0.5).astype(int)
    else:
        return [int(np.floor(val + 0.5)) for val in x]

def uniquer(particle):
    """Garante que cada partícula tenha valores únicos."""
    unique_particle = np.unique(particle)
    if len(unique_particle) < nvar:
        missing = nvar - len(unique_particle)
        new_values = np.random.uniform(-5.12, 5.12, size=missing)
        unique_particle = np.concatenate([unique_particle, new_values])
    return unique_particle[:nvar]

# -------------------------------
# Inicialização da população
# -------------------------------
particula = np.random.uniform(-5.12, 5.12, size=nvar)
global_best = particula.copy()
Particulas = np.tile(particula, (npar, 1)).copy()

Weighta = np.ones((npar, nvar))
Weightb = np.ones((npar, nvar))
Weightc = np.full((npar, nvar), 2.0)

a = np.zeros((npar, nvar))
b = np.zeros((npar, nvar))
c = np.zeros((npar, nvar))

# Redução linear do fator de inércia
w = 2 - ((2.2 - 0.4) / max_it) * (np.arange(max_it) / max_it)

Personal_bests = Particulas.copy()
Fitness_Personal_Bests = np.full(npar, np.inf)
Fitness_global_best = np.inf

Velocities = np.ones((npar, nvar))
best_cost_iteration = np.zeros(max_it)

# ============================================================
# Laço principal do EPSO
# ============================================================
for it in range(max_it):

    # Replicação
    Particulas = np.tile(particula, (npar, 1)).copy()
    Eval = np.full(npar, np.inf)

    # Mutação dos pesos
    Na = np.random.normal(0, 1, (npar, nvar))
    Nb = np.random.normal(0, 1, (npar, nvar))
    Nc = np.random.normal(0, 1, (npar, nvar))

    Weighta = Na * T * w[it] * Weighta
    Weightb = Weightb * Nb * T
    Weightc = Weightc * Nc * T

    # Reprodução (atualização da velocidade)
    a = Weighta * Velocities
    b = Weightb * (Personal_bests - Particulas)
    c = Weightc * (global_best - Particulas)

    Velocities = a + b + c
    Velocities = np.clip(Velocities, -1.0, 1.0)  # Limita a variação

    # Atualização da posição
    Particulas = Particulas + Velocities
    Particulas = np.clip(Particulas, -5.12, 5.12)

    # Garante unicidade (mesma ideia do código original)
    for i in range(npar):
        Particulas[i] = uniquer(Particulas[i])

    # Avaliação (Rastrigin)
    for i in range(npar):
        Eval[i] = rastrigin(Particulas[i])

    # Seleção: atualiza melhores pessoais e globais
    improved_mask = Eval < Fitness_Personal_Bests
    Fitness_Personal_Bests[improved_mask] = Eval[improved_mask]

    for i in np.where(improved_mask)[0]:
        Personal_bests[i] = Particulas[i].copy()

    min_index = np.argmin(Fitness_Personal_Bests)
    if Fitness_Personal_Bests[min_index] < Fitness_global_best:
        Fitness_global_best = Fitness_Personal_Bests[min_index]
        global_best = Personal_bests[min_index].copy()

    particula = global_best.copy()
    best_cost_iteration[it] = Fitness_global_best

    if it % 50 == 0:
        print(f"It {it}: melhor valor = {Fitness_global_best:.6f}")

# ============================================================
# Resultados
# ============================================================
print("\nMelhor solução encontrada:")
print(global_best)
print(f"Melhor valor da função: {Fitness_global_best:.6f}")

# Curva de convergência
plt.figure(figsize=(8,5))
plt.plot(best_cost_iteration, label="Melhor valor (EPSO)")
plt.xlabel("Iterações")
plt.ylabel("Função de custo")
plt.title("Convergência do EPSO — Função de Rastrigin")
plt.grid(True)
plt.legend()
plt.show()
