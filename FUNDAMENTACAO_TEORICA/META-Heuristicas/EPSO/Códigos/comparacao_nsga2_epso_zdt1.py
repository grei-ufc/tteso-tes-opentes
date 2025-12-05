import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.problems.multi.zdt import ZDT1
from pymoo.termination import get_termination
from pymoo.indicators.igd import IGD

# ============================================================
# 1. DEFINIÇÃO DO PROBLEMA ZDT1
# ============================================================

problem = ZDT1(n_var=30)  # Problema clássico multiobjetivo com 30 variáveis

# ============================================================
# 2. CLASSE EPSO MULTIOBJETIVO
# ============================================================

class EPSO_MO:
    """
    Implementação simples de um Enriched PSO (EPSO) para otimização multiobjetivo.
    Baseado na ideia de usar um repositório (arquivo) de soluções não dominadas.
    """

    def __init__(self, problem, swarm_size=100, max_iter=200):
        self.problem = problem
        self.n_var = problem.n_var
        self.swarm_size = swarm_size
        self.max_iter = max_iter

        # Limites do problema — correção principal aqui 👇
        self.xl = np.array(problem.xl)
        self.xu = np.array(problem.xu)

        # Inicialização
        self.pos = np.random.rand(swarm_size, self.n_var)
        self.pos = self.xl + self.pos * (self.xu - self.xl)
        self.vel = np.zeros((swarm_size, self.n_var))

        # Avaliação inicial
        self.fitness = problem.evaluate(self.pos)
        self.archive = self.pos.copy()
        self.archive_fit = self.fitness.copy()

    # ----------------------------------------------------------
    def dominates(self, a, b):
        """Verifica se solução 'a' domina solução 'b'."""
        return np.all(a <= b) and np.any(a < b)

    # ----------------------------------------------------------
    def update_archive(self):
        """Atualiza o repositório de soluções não dominadas."""
        new_archive = []
        new_fit = []

        for i in range(len(self.archive)):
            dominated = False
            for j in range(len(self.archive)):
                if i != j and self.dominates(self.archive_fit[j], self.archive_fit[i]):
                    dominated = True
                    break
            if not dominated:
                new_archive.append(self.archive[i])
                new_fit.append(self.archive_fit[i])

        self.archive = np.array(new_archive)
        self.archive_fit = np.array(new_fit)

    # ----------------------------------------------------------
    def run(self):
        """Executa o EPSO multiobjetivo."""
        w = 0.6  # inércia
        c1 = 1.5  # peso cognitivo
        c2 = 1.5  # peso social

        for t in range(self.max_iter):
            # Atualização das velocidades e posições
            r1 = np.random.rand(self.swarm_size, self.n_var)
            r2 = np.random.rand(self.swarm_size, self.n_var)

            # Escolhe aleatoriamente líderes do arquivo (soluções não dominadas)
            leaders = self.archive[np.random.randint(0, len(self.archive), size=self.swarm_size)]

            self.vel = w * self.vel + c1 * r1 * (self.archive - self.pos) + c2 * r2 * (leaders - self.pos)
            self.pos = self.pos + self.vel

            # Mantém dentro dos limites corrigidos 👇
            self.pos = np.clip(self.pos, self.xl, self.xu)

            # Avalia novas soluções
            self.fitness = self.problem.evaluate(self.pos)

            # Combina e atualiza arquivo
            self.archive = np.vstack((self.archive, self.pos))
            self.archive_fit = np.vstack((self.archive_fit, self.fitness))
            self.update_archive()

            if (t + 1) % 50 == 0:
                print(f"Iteração {t+1}/{self.max_iter} | Arquivo: {len(self.archive)} soluções")

        return self.archive, self.archive_fit


# ============================================================
# 3. EXECUÇÃO DO NSGA-II (referência)
# ============================================================

termination = get_termination("n_gen", 200)
nsga2 = NSGA2(pop_size=100)
res_nsga2 = minimize(problem, nsga2, termination, seed=42, verbose=False)

# ============================================================
# 4. EXECUÇÃO DO EPSO
# ============================================================

epso = EPSO_MO(problem, swarm_size=100, max_iter=200)
epso_solutions, epso_objectives = epso.run()

# ============================================================
# 5. MÉTRICAS DE DESEMPENHO (IGD)
# ============================================================

igd_metric = IGD(res_nsga2.F)
igd_epso = igd_metric(epso_objectives)
igd_nsga2 = igd_metric(res_nsga2.F)

print("\n=== RESULTADOS DE DESEMPENHO ===")
print(f"IGD EPSO  = {igd_epso:.5f}")
print(f"IGD NSGA2 = {igd_nsga2:.5f}")

# ============================================================
# 6. PLOTAGEM DO FRENTE DE PARETO
# ============================================================

plt.figure(figsize=(8, 6))
plt.scatter(res_nsga2.F[:, 0], res_nsga2.F[:, 1], s=20, label="NSGA-II", alpha=0.7)
plt.scatter(epso_objectives[:, 0], epso_objectives[:, 1], s=20, label="EPSO", alpha=0.7)
plt.title("Comparação entre NSGA-II e EPSO no problema ZDT1")
plt.xlabel("F1")
plt.ylabel("F2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
