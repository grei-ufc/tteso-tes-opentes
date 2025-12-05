import numpy as np

# ---------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------

def mostrar_vetor(vetor, colunas=8):
    """Exibe um vetor formatado em múltiplas colunas."""
    for i, valor in enumerate(vetor):
        if i % colunas == 0:
            print("\n", end="")
        print(f"{valor:8.4f}", end=" ")
    print("\n")

def rastrigin(posicao):
    """
    Função de Rastrigin: benchmark comum em otimização.
    Mínimo global = 0 no ponto [0, 0, ..., 0].
    """
    posicao = np.array(posicao)
    return np.sum(posicao**2 - 10 * np.cos(2 * np.pi * posicao) + 10)

# ---------------------------------------------------
# Classe Partícula
# ---------------------------------------------------

class Particula:
    def __init__(self, dimensao, minx, maxx, rng):
        self.posicao = rng.uniform(minx, maxx, dimensao)
        self.velocidade = rng.uniform(minx, maxx, dimensao)
        self.melhor_posicao = np.copy(self.posicao)
        self.erro = rastrigin(self.posicao)
        self.melhor_erro = self.erro

# ---------------------------------------------------
# Algoritmo PSO
# ---------------------------------------------------

def pso(max_epocas, n_particulas, dimensao, minx, maxx):
    rng = np.random.default_rng(seed=42)  # gerador aleatório moderno

    # inicializa o enxame
    enxame = [Particula(dimensao, minx, maxx, rng) for _ in range(n_particulas)]

    # melhor solução global até agora
    melhor_enxame_pos = np.copy(enxame[0].posicao)
    melhor_enxame_erro = enxame[0].erro

    for particula in enxame:
        if particula.erro < melhor_enxame_erro:
            melhor_enxame_erro = particula.erro
            melhor_enxame_pos = np.copy(particula.posicao)

    # hiperparâmetros do PSO (valores padrão sugeridos)
    w = 0.729    # inércia
    c1 = 1.49445 # componente cognitiva (partícula)
    c2 = 1.49445 # componente social (enxame)

    # loop principal
    for epoca in range(max_epocas):
        if epoca % 10 == 0 and epoca > 0:
            print(f"Época {epoca:3d} | Melhor erro = {melhor_enxame_erro:.6f}")

        for particula in enxame:
            r1 = rng.random(dimensao)
            r2 = rng.random(dimensao)

            # atualização da velocidade
            particula.velocidade = (
                w * particula.velocidade
                + c1 * r1 * (particula.melhor_posicao - particula.posicao)
                + c2 * r2 * (melhor_enxame_pos - particula.posicao)
            )

            # atualização da posição
            particula.posicao += particula.velocidade

            # aplica limites (clamp)
            particula.posicao = np.clip(particula.posicao, minx, maxx)

            # avalia novo erro
            particula.erro = rastrigin(particula.posicao)

            # atualiza melhor posição individual
            if particula.erro < particula.melhor_erro:
                particula.melhor_erro = particula.erro
                particula.melhor_posicao = np.copy(particula.posicao)

            # atualiza melhor posição global
            if particula.erro < melhor_enxame_erro:
                melhor_enxame_erro = particula.erro
                melhor_enxame_pos = np.copy(particula.posicao)

    return melhor_enxame_pos, melhor_enxame_erro

# ---------------------------------------------------
# Execução do algoritmo
# ---------------------------------------------------

if __name__ == "__main__":
    print("\n=== Otimização por Enxame de Partículas (PSO) ===\n")
    dimensao = 3
    n_particulas = 50
    max_epocas = 100

    print(f"Função: Rastrigin em {dimensao} variáveis")
    print("Mínimo conhecido = 0.0 no ponto (0, 0, ..., 0)\n")
    print(f"Configuração: {n_particulas} partículas, {max_epocas} épocas\n")

    melhor_pos, melhor_erro = pso(max_epocas, n_particulas, dimensao, -10.0, 10.0)

    print("\n=== Resultado Final ===")
    print("Melhor solução encontrada:")
    mostrar_vetor(melhor_pos)
    print(f"Erro da melhor solução = {melhor_erro:.6f}")
    print("\nFim da execução do PSO\n")
