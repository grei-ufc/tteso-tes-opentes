import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIGURAÇÕES DO EPSO
# ------------------------------------------------------------
npar = 10      # número de partículas
nvar = 7       # tamanho da partícula (número de nós)
T = 1          # taxa de mutação
max_it = 300   # número de iterações

# ------------------------------------------------------------
# FUNÇÃO DE AVALIAÇÃO GENÉRICA (substitui run_opt)
# ------------------------------------------------------------
def evaluate_particle(particle):
    # Função de custo: minimiza a distância até o valor 50
    return np.sum((particle - 50)**2)

# ------------------------------------------------------------
# FUNÇÃO AUXILIAR DE ARREDONDAMENTO
# ------------------------------------------------------------
def round_half_up(x):
    if isinstance(x, (int, float)):
        return int(np.floor(x + 0.5))
    elif isinstance(x, np.ndarray):
        return np.floor(x + 0.5).astype(int)
    else:
        return [int(np.floor(val + 0.5)) for val in x]

# ------------------------------------------------------------
# FUNÇÃO DE UNICIDADE (evita nós repetidos)
# ------------------------------------------------------------
def uniquer(particle):
    unique_particle = np.unique(particle)
    if len(unique_particle) < nvar:
        missing = nvar - len(unique_particle)
        new_values = np.random.randint(7, 75, size=missing)
        unique_particle = np.concatenate([unique_particle, new_values])
    return unique_particle[:nvar]

# ------------------------------------------------------------
# INICIALIZAÇÃO
# ------------------------------------------------------------
particula = np.random.randint(7, 75, size=nvar)
global_best = particula.copy()
Particulas = np.tile(particula, (npar, 1)).copy()

Weighta = np.ones((npar, nvar))
Weightb = np.ones((npar, nvar))
Weightc = np.full((npar, nvar), 2.0)

Velocities = np.ones((npar, nvar))
Personal_bests = Particulas.copy()
Fitness_Personal_Bests = np.full(npar, np.inf)
Fitness_global_best = np.inf

w = 2 - ((2.2 - 0.4) / max_it) * (np.arange(max_it) / max_it)
best_cost_iteration = np.zeros(max_it)

# ------------------------------------------------------------
# LOOP PRINCIPAL DO EPSO
# ------------------------------------------------------------
for it in range(max_it):

    # Replicação
    Particulas = np.tile(particula, (npar, 1)).copy()
    Eval = np.full(npar, np.inf)
    
    # Mutação
    Na = np.random.normal(0, 1, (npar, nvar))
    Nb = np.random.normal(0, 1, (npar, nvar))
    Nc = np.random.normal(0, 1, (npar, nvar))
    Weighta = Na * T * w[it] * Weighta
    Weightb = Weightb * Nb * T
    Weightc = Weightc * Nc * T
    
    # Reprodução
    a = Weighta * Velocities
    b = Weightb * (Personal_bests - Particulas)
    c = Weightc * (global_best - Particulas)
    Velocities = a + b + c
    Velocities = np.clip(Velocities, -7.1, 7.1)
    
    # Atualização das partículas
    Particulas = round_half_up(Particulas + Velocities).astype(int)
    Particulas = np.clip(Particulas, 7, 74)

    # Garante unicidade
    for i in range(npar):
        Particulas[i] = uniquer(Particulas[i])
    
    # Avaliação
    for i in range(npar):
        Eval[i] = evaluate_particle(Particulas[i])
    
    # Seleção
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

# ------------------------------------------------------------
# RESULTADOS
# ------------------------------------------------------------
print("Melhor valor encontrado:", Fitness_global_best)
print("Melhor partícula (configuração de nós):", global_best)

plt.plot(best_cost_iteration)
plt.xlabel("Iterações")
plt.ylabel("Melhor Fitness Global")
plt.title("Convergência do EPSO (função genérica)")
plt.grid(True)
plt.show()
