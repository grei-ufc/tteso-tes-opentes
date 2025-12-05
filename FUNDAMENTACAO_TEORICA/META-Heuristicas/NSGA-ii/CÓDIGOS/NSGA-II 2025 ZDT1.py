import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIGURAÇÕES
# ------------------------------------------------------------
pop_size = 100
nvar = 30
max_gen = 200

# ------------------------------------------------------------
# FUNÇÃO ZDT1
# ------------------------------------------------------------
def zdt1(ind):
    f1 = ind[0]
    g = 1 + 9 * np.sum(ind[1:]) / (nvar - 1)
    f2 = g * (1 - np.sqrt(f1 / g))
    return f1, f2

# ------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ------------------------------------------------------------
def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)

def non_dominated_sort(F):
    fronts = []
    S = [[] for _ in range(len(F))]
    n = np.zeros(len(F), dtype=int)
    rank = np.zeros(len(F), dtype=int)

    for p in range(len(F)):
        for q in range(len(F)):
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
    current_front = [i for i in range(len(F)) if rank[i] == 0]
    fronts.append(current_front)
    
    while current_front:
        next_front = []
        for p in current_front:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = len(fronts)
                    next_front.append(q)
        current_front = next_front
        if current_front:
            fronts.append(current_front)
    return fronts

# ------------------------------------------------------------
# AVALIAÇÃO POPULAÇÃO
# ------------------------------------------------------------
def evaluate_population(pop):
    f_vals = np.array([zdt1(ind) for ind in pop])
    return f_vals

# ------------------------------------------------------------
# INICIALIZA POPULAÇÃO
# ------------------------------------------------------------
pop = np.random.rand(pop_size, nvar)

# ------------------------------------------------------------
# LOOP PRINCIPAL NSGA-II
# ------------------------------------------------------------
for gen in range(max_gen):
    # Reprodução simples (mutação + cruzamento)
    offspring = pop + np.random.normal(0, 0.1, size=pop.shape)
    offspring = np.clip(offspring, 0, 1)
    combined = np.vstack((pop, offspring))
    F = evaluate_population(combined)
    fronts = non_dominated_sort(F)

    # Seleção por fronts
    new_pop = []
    for front in fronts:
        for i in front:
            new_pop.append(combined[i])
            if len(new_pop) >= pop_size:
                break
        if len(new_pop) >= pop_size:
            break
    pop = np.array(new_pop)

# ------------------------------------------------------------
# RESULTADOS
# ------------------------------------------------------------
F_final = evaluate_population(pop)
plt.scatter(F_final[:, 0], F_final[:, 1], c="blue", s=20)
plt.xlabel("f1(x)")
plt.ylabel("f2(x)")
plt.title("Fronteira de Pareto - ZDT1 (NSGA-II)")
plt.grid(True)
plt.show()
