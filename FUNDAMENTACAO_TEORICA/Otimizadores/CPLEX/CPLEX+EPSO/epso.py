import numpy as np

class EPSO:
    def __init__(self, npart=20, ngen=30, T=6):
        self.npart = npart
        self.ngen = ngen
        self.T = T

    def optimize(self, fitness_function):
        particles = np.random.uniform(-2, 2, (self.npart, self.T))
        best_particle = None
        best_cost = 1e12

        for gen in range(self.ngen):
            # Mutação
            mutated = particles + 0.3 * np.random.randn(*particles.shape)

            # Combinação
            offspring = 0.5 * (particles + mutated)

            # Avaliação fitness via CPLEX
            costs = np.array([fitness_function(p) for p in offspring])

            # Seleção dos melhores
            idx = np.argsort(costs)
            particles = offspring[idx[:self.npart]]

            # Atualiza melhor global
            if costs[idx[0]] < best_cost:
                best_cost = costs[idx[0]]
                best_particle = particles[0].copy()

            print(f"Geração {gen:02d} | Melhor custo = {best_cost:.2f}")

        return best_particle, best_cost
