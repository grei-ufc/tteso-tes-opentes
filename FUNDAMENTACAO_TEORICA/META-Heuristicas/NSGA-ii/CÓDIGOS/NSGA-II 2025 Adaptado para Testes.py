import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIGURAÇÕES
# ------------------------------------------------------------
pop_size = 50
nvar = 7
max_gen = 200

# ------------------------------------------------------------
# FUNÇÕES OBJETIVO
# ------------------------------------------------------------
def f1(x):  # Quadrático
    return np.sum((x - 50)**2)

def f2(x):  # Linear absoluto
    return np.sum(np.abs(x - 25))

# ------------------------------------------------------------
# INICIALIZA POPULAÇÃO
# ------------------------------------------------------------
pop = np.random.randint(7, 75, (pop_size, nvar))

# ------------------------------------------------------------
# AVALIAÇÃO
# ------------------------------------------------------------
def evaluate_population(pop):
    f1_vals = np.array([f1(ind) for ind in pop])
    f2_vals = np.array([f2(ind) for ind in pop])
    return np.vstack((f1_vals, f2_vals)).T

# ------------------------------------------------------------
# NON-DOMINATED SORTING
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
# LOOP DO NSGA-II
# ------------------------------------------------------------
for gen in range(max_gen):
    offspring = np.clip(
        pop + np.random.randint(-5, 6, size=pop.shape), 7, 74
    )
    combined = np.vstack((pop, offspring))
    F = evaluate_population(combined)
    fronts = non_dominated_sort(F)

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
# PLOT
# ------------------------------------------------------------
F_final = evaluate_population(pop)
plt.scatter(F_final[:, 0], F_final[:, 1], c="blue")
plt.xlabel("f1(x) = Σ(x−50)²")
plt.ylabel("f2(x) = Σ|x−25|")
plt.title("Fronteira de Pareto aproximada — NSGA-II simplificado")
plt.grid(True)
plt.show()
