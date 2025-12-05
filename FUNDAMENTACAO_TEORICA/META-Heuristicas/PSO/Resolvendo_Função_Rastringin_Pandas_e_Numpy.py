import numpy as np
import pandas as pd

# ----------------------------
# Função de teste: Rastrigin
# ----------------------------
def rastrigin(populacao: np.ndarray) -> np.ndarray:
    """
    Calcula a função de Rastrigin para uma população de partículas.
    populacao: array (n_particulas, dimensao)
    retorna: array (n_particulas,) com o valor de cada partícula
    """
    return np.sum(populacao**2 - 10 * np.cos(2 * np.pi * populacao) + 10, axis=1)


# ----------------------------
# Algoritmo PSO
# ----------------------------
def pso(funcao_objetivo,
        dimensao: int,
        minimo: float,
        maximo: float,
        n_particulas: int = 50,
        max_epocas: int = 200,
        w: float = 0.729,
        c1: float = 1.49445,
        c2: float = 1.49445,
        vmax_frac: float = 0.2,
        semente: int | None = None,
        verbose: bool = True):
    """
    Implementação moderna do PSO.
    """
    rng = np.random.default_rng(semente)

    # Inicialização
    posicoes = rng.uniform(minimo, maximo, size=(n_particulas, dimensao))
    vmax = (maximo - minimo) * vmax_frac
    velocidades = rng.uniform(-vmax, vmax, size=(n_particulas, dimensao))

    valores = funcao_objetivo(posicoes)

    melhor_pos_pessoal = posicoes.copy()
    melhor_valor_pessoal = valores.copy()

    idx_melhor_global = np.argmin(melhor_valor_pessoal)
    melhor_pos_global = melhor_pos_pessoal[idx_melhor_global].copy()
    melhor_valor_global = melhor_valor_pessoal[idx_melhor_global]

    historico = []

    # Loop principal
    for epoca in range(max_epocas):
        r1 = rng.random((n_particulas, dimensao))
        r2 = rng.random((n_particulas, dimensao))

        # Atualização de velocidades
        velocidades = (w * velocidades
                       + c1 * r1 * (melhor_pos_pessoal - posicoes)
                       + c2 * r2 * (melhor_pos_global - posicoes))

        # Limitando velocidades
        velocidades = np.clip(velocidades, -vmax, vmax)

        # Atualização de posições
        posicoes += velocidades
        posicoes = np.clip(posicoes, minimo, maximo)

        # Avaliar nova posição
        valores = funcao_objetivo(posicoes)

        # Atualização de melhores pessoais
        melhora = valores < melhor_valor_pessoal
        melhor_pos_pessoal[melhora] = posicoes[melhora]
        melhor_valor_pessoal[melhora] = valores[melhora]

        # Atualização de melhor global
        idx = np.argmin(melhor_valor_pessoal)
        if melhor_valor_pessoal[idx] < melhor_valor_global:
            melhor_valor_global = melhor_valor_pessoal[idx]
            melhor_pos_global = melhor_pos_pessoal[idx].copy()

        # Guardar histórico
        historico.append({
            "época": epoca,
            "melhor_valor_global": melhor_valor_global
        })

        if verbose and epoca % 10 == 0:
            print(f"Época {epoca:4d} | Melhor valor global = {melhor_valor_global:.6f}")

    # Converter histórico em DataFrame pandas
    df_historico = pd.DataFrame(historico)

    return melhor_pos_global, melhor_valor_global, df_historico


# ----------------------------
# Exemplo de uso
# ----------------------------
if __name__ == "__main__":
    dimensao = 3
    n_particulas = 50
    max_epocas = 100

    print("\nInício da demonstração do PSO moderno (com NumPy + pandas)\n")
    print(f"Objetivo: minimizar a função de Rastrigin em {dimensao} variáveis")
    print("Mínimo conhecido = 0.0 em (0, 0, ..., 0)\n")

    melhor_pos, melhor_valor, historico = pso(rastrigin,
                                              dimensao,
                                              minimo=-5.12,
                                              maximo=5.12,
                                              n_particulas=n_particulas,
                                              max_epocas=max_epocas,
                                              semente=42,
                                              verbose=True)

    print("\nPSO concluído\n")
    print("Melhor posição encontrada:", melhor_pos)
    print("Melhor valor encontrado:", melhor_valor)

    # Salvar histórico em CSV
    historico.to_csv("historico_pso.csv", index=False)
    print("\nHistórico de convergência salvo em 'historico_pso.csv'\n")
