import random
import numpy as np
from docplex.mp.model import Model
import matplotlib.pyplot as plt


# ============================================================
# 1 – Função de avaliação mais complexa
# ============================================================
def avaliar_solucao(parametros):
    T = 24
    dim = len(parametros)

    w = np.clip(parametros, 0, 1)

    m = Model()

    pv = [max(0, 5 - abs(t - 12)) for t in range(T)]
    demanda = [2 + 0.5*np.sin(t) for t in range(T)]

    grid = m.continuous_var_list(T, lb=0)
    carga_bat = m.continuous_var_list(T, lb=0)
    des_bat = m.continuous_var_list(T, lb=0)
    SOC = m.continuous_var_list(T, lb=0, ub=10)

    eta_c = 0.9
    eta_d = 0.9

    for t in range(T):
        peso = w[t % dim]
        prod_pv = pv[t] * (1 + 0.1 * peso**2)

        m.add_constraint(
            grid[t] + prod_pv + des_bat[t] * eta_d ==
            demanda[t] + carga_bat[t]
        )

    m.add_constraint(SOC[0] == 5)
    for t in range(1, T):
        m.add_constraint(
            SOC[t] == SOC[t-1] + carga_bat[t]*eta_c - des_bat[t]/eta_d
        )

    custo = m.sum(grid[t]*(0.7 + 0.05*np.random.randn()) for t in range(T))
    penal_soc = m.sum((1 - SOC[t]/10)**2 for t in range(T))

    m.minimize(custo + 0.1 * penal_soc)

    sol = m.solve()
    return sol.objective_value if sol else 1e9



# ============================================================
# 2 – PSO
# ============================================================
def PSO(num_part=50, dim=24, max_iter=60):
    particulas = [np.random.uniform(0, 1, dim) for _ in range(num_part)]
    velocidades = [np.zeros(dim) for _ in range(num_part)]
    pb = particulas.copy()
    pb_fitness = [avaliar_solucao(p) for p in particulas]

    gb = pb[np.argmin(pb_fitness)]
    gb_f = min(pb_fitness)

    w, c1, c2 = 0.5, 1.4, 1.4
    historico = []

    for it in range(max_iter):
        for i in range(num_part):

            velocidades[i] = (
                w * velocidades[i]
                + c1 * random.random() * (pb[i] - particulas[i])
                + c2 * random.random() * (gb - particulas[i])
            )

            particulas[i] += velocidades[i]
            particulas[i] = np.clip(particulas[i], 0, 1)

            f = avaliar_solucao(particulas[i])

            if f < pb_fitness[i]:
                pb[i], pb_fitness[i] = particulas[i].copy(), f

            if f < gb_f:
                gb, gb_f = particulas[i].copy(), f

        # guarda o **melhor valor até agora**
        historico.append(gb_f)

        print(f"PSO iter {it}: {gb_f:.2f}")

    return gb, gb_f, historico



# ============================================================
# 3 – Algoritmo Genético
# ============================================================
def GA(pop=50, dim=24, iter=60, mut_rate=0.15):
    populacao = [np.random.uniform(0, 1, dim) for _ in range(pop)]
    fitness = [avaliar_solucao(ind) for ind in populacao]

    melhor_global = min(fitness)
    historico = [melhor_global]

    for it in range(iter):

        pais = sorted(zip(populacao, fitness), key=lambda x: x[1])[:pop//2]
        nova_pop = []

        while len(nova_pop) < pop:
            p1, _ = random.choice(pais)
            p2, _ = random.choice(pais)

            cx = (p1 + p2)/2

            if random.random() < mut_rate:
                cx += np.random.normal(0, 0.2, dim)

            nova_pop.append(np.clip(cx, 0, 1))

        populacao = nova_pop
        fitness = [avaliar_solucao(ind) for ind in populacao]

        melhor_iter = min(fitness)

        if melhor_iter < melhor_global:
            melhor_global = melhor_iter

        historico.append(melhor_global)

        print(f"GA iter {it}: {melhor_iter:.2f}")

    melhor_ind = populacao[np.argmin(fitness)]
    return melhor_ind, melhor_global, historico



# ============================================================
# 4 – EPSO com mutação adaptativa
# ============================================================
def EPSO(num_part=50, dim=24, max_iter=60):
    particulas = [np.random.uniform(0,1,dim) for _ in range(num_part)]
    fitness = [avaliar_solucao(p) for p in particulas]

    melhor_global = min(fitness)
    historico = [melhor_global]

    w = np.ones(dim)*0.4

    for it in range(max_iter):
        filhos = []

        for p in particulas:
            filho = (
                p
                + w * np.random.normal(0, 0.2, dim)
                + np.random.uniform(-0.15, 0.15, dim)
            )
            filhos.append(np.clip(filho, 0, 1))

        todos = particulas + filhos
        fitness_todos = [avaliar_solucao(x) for x in todos]

        idx = np.argsort(fitness_todos)[:num_part]
        particulas = [todos[i] for i in idx]
        fitness = [fitness_todos[i] for i in idx]

        melhor_iter = min(fitness)
        if melhor_iter < melhor_global:
            melhor_global = melhor_iter

        historico.append(melhor_global)

        w += np.random.normal(0, 0.05, dim)
        w = np.clip(w, 0.1, 1.0)

        print(f"EPSO iter {it}: {melhor_iter:.2f}")

    melhor = particulas[np.argmin(fitness)]
    return melhor, melhor_global, historico



# ============================================================
# 5 – Rodar tudo
# ============================================================
pso_sol, pso_f, hist_pso = PSO()
ga_sol, ga_f, hist_ga = GA()
epso_sol, epso_f, hist_epso = EPSO()

print("\n===== RESULTADOS =====")
print(f"PSO  => custo = {pso_f:.2f}")
print(f"GA   => custo = {ga_f:.2f}")
print(f"EPSO => custo = {epso_f:.2f}")


# ============================================================
# 6 – Gráfico comparativo
# ============================================================
plt.plot(hist_pso, label="PSO")
plt.plot(hist_ga, label="GA")
plt.plot(hist_epso, label="EPSO")

plt.xlabel("Iteração")
plt.ylabel("Custo (melhor até agora)")
plt.title("Comparativo – PSO x GA x EPSO")
plt.legend()
plt.grid()
plt.show()
