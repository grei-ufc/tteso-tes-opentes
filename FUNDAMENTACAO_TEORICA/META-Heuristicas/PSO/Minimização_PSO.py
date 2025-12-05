import numpy as np

# -------------------------
# Funções de restrição / penalidade
# -------------------------
def penalized_linear_objective(pop, c, A, b, sense, penalty=1e6):
    """
    pop: array (n, dim) ou (dim,) -> população de pontos
    c: vetor coeficientes (dim,)
    A: array (m, dim) com m restrições
    b: vetor (m,)
    sense: lista/array de strings com 'le', 'ge' ou 'eq' para cada restrição
    penalty: coeficiente de penalidade (grande)
    Retorna: vetor de custos penalizados (n,)
    """
    pop_arr = np.atleast_2d(pop)  # shape (N, dim)
    n = pop_arr.shape[0]
    obj = pop_arr.dot(c)  # (N,)
    viol = np.zeros(n)

    for i, Ai in enumerate(A):
        lhs = pop_arr.dot(Ai)
        if sense[i] == 'le':   # A x <= b  -> viol = max(0, Ax - b)
            viol += np.maximum(0.0, lhs - b[i])
        elif sense[i] == 'ge': # A x >= b  -> viol = max(0, b - Ax)
            viol += np.maximum(0.0, b[i] - lhs)
        elif sense[i] == 'eq': # A x == b  -> viol = abs(Ax - b)
            viol += np.abs(lhs - b[i])
        else:
            raise ValueError("sense deve conter 'le','ge' ou 'eq'.")

    return obj + penalty * viol


def check_constraints(x, A, b, sense, tol=1e-8):
    """Retorna True se x satisfaz todas as restrições dentro de tol."""
    x = np.asarray(x)
    for i, Ai in enumerate(A):
        val = Ai.dot(x)
        if sense[i] == 'le':
            if val > b[i] + tol:
                return False
        elif sense[i] == 'ge':
            if val < b[i] - tol:
                return False
        elif sense[i] == 'eq':
            if abs(val - b[i]) > tol:
                return False
    return True


# -------------------------
# PSO vetorizado com penalidade
# -------------------------
def pso_linear_penalty(c, A, b, sense,
                       dim,
                       minx=0.0, maxx=10.0,
                       n_particles=50, max_iter=200,
                       w=0.729, c1=1.49445, c2=1.49445,
                       vmax_frac=0.2, penalty=1e6,
                       seed=42, verbose=True):
    rng = np.random.default_rng(seed)

    # inicializa posições e velocidades
    pos = rng.uniform(minx, maxx, size=(n_particles, dim))
    vmax = (maxx - minx) * vmax_frac
    vel = rng.uniform(-vmax, vmax, size=(n_particles, dim))

    # avalia com penalidade
    penal = lambda P: penalized_linear_objective(P, c, A, b, sense, penalty=penalty)
    scores = penal(pos)  # (n_particles,)

    pbest_pos = pos.copy()
    pbest_scores = scores.copy()

    g_idx = np.argmin(pbest_scores)
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_score = pbest_scores[g_idx]

    history = []

    for it in range(max_iter):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))

        vel = (w * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (gbest_pos - pos))

        vel = np.clip(vel, -vmax, vmax)
        pos += vel
        pos = np.clip(pos, minx, maxx)

        scores = penal(pos)

        improved = scores < pbest_scores
        if np.any(improved):
            pbest_pos[improved] = pos[improved]
            pbest_scores[improved] = scores[improved]

        g_idx = np.argmin(pbest_scores)
        if pbest_scores[g_idx] < gbest_score:
            gbest_score = pbest_scores[g_idx]
            gbest_pos = pbest_pos[g_idx].copy()

        # guardar histórico (mostramos o custo *penalizado* e também o valor real sem penalidade)
        true_value = np.dot(gbest_pos, c)
        history.append((it, gbest_score, true_value))

        if verbose and (it % 10 == 0):
            print(f"Época {it:4d} | Melhor (com penalidade) = {gbest_score:.6f} | Valor verdadeiro = {true_value:.6f}")

    return gbest_pos, gbest_score, history


# -------------------------
# Exemplo: problema que leva a (0,3)
# -------------------------
if __name__ == "__main__":
    print("\n=== PSO para Otimização Linear (com penalidade) ===\n")

    # Coeficientes da função objetivo f(x) = c^T x
    c = np.array([3.0, 2.0])  # 3*x1 + 2*x2

    # Restrição que impede (0,0): x1 + x2 >= 3
    A = np.array([[1.0, 1.0]])
    b = np.array([3.0])
    sense = ['ge']  # 'ge' = >=

    dim = len(c)
    n_particles = 40
    max_iter = 100
    minx, maxx = 0.0, 10.0

    best_pos, best_penalized_score, hist = pso_linear_penalty(
        c, A, b, sense,
        dim,
        minx=minx, maxx=maxx,
        n_particles=n_particles, max_iter=max_iter,
        seed=42,
        verbose=True
    )

    print("\n=== Resultado Final ===")
    print("Melhor posição (vetor):", np.round(best_pos, 6))
    print("Satisfaz restrições?", check_constraints(best_pos, A, b, sense))
    print("Valor da função objetivo (sem penalidade) = {:.6f}".format(np.dot(best_pos, c)))
    print("\nFim da execução\n")
