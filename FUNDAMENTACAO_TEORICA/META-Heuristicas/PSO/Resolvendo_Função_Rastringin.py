# Demonstração do algoritmo de Otimização por Enxame de Partículas (PSO)
# Resolve a função de Rastrigin

import random
import math    # cos() para a função de Rastrigin
import copy    # conveniência para copiar arrays
import sys     # para usar o maior float possível

# ------------------------------------

def mostrar_vetor(vetor):
    for i in range(len(vetor)):
        if i % 8 == 0:  # 8 colunas por linha
            print("\n", end="")
        if vetor[i] >= 0.0:
            print(' ', end="")
        print("%.4f" % vetor[i], end="")  # 4 casas decimais
        print(" ", end="")
    print("\n")

def erro(posicao):
    soma = 0.0
    for i in range(len(posicao)):
        xi = posicao[i]
        soma += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return soma

# ------------------------------------

class Particula:
    def __init__(self, dimensao, minimo, maximo, semente):
        self.rnd = random.Random(semente)
        self.posicao = [0.0 for i in range(dimensao)]
        self.velocidade = [0.0 for i in range(dimensao)]
        self.melhor_pos_particula = [0.0 for i in range(dimensao)]

        for i in range(dimensao):
            self.posicao[i] = ((maximo - minimo) *
                               self.rnd.random() + minimo)
            self.velocidade[i] = ((maximo - minimo) *
                                  self.rnd.random() + minimo)

        self.erro = erro(self.posicao)  # erro atual
        self.melhor_pos_particula = copy.copy(self.posicao)
        self.melhor_erro_particula = self.erro  # melhor erro pessoal

# ------------------------------------

def resolver(max_epocas, n, dimensao, minimo, maximo):
    rnd = random.Random(0)

    # cria n partículas aleatórias
    enxame = [Particula(dimensao, minimo, maximo, i) for i in range(n)]

    melhor_pos_enxame = [0.0 for i in range(dimensao)]
    melhor_erro_enxame = sys.float_info.max  # melhor global

    # verifica cada partícula inicial
    for i in range(n):
        if enxame[i].erro < melhor_erro_enxame:
            melhor_erro_enxame = enxame[i].erro
            melhor_pos_enxame = copy.copy(enxame[i].posicao)

    epoca = 0
    w = 0.729    # inércia
    c1 = 1.49445 # fator cognitivo (partícula)
    c2 = 1.49445 # fator social (enxame)

    while epoca < max_epocas:

        if epoca % 10 == 0 and epoca > 1:
            print("Época = " + str(epoca) +
                  " melhor erro = %.3f" % melhor_erro_enxame)

        for i in range(n):  # processa cada partícula

            # calcula nova velocidade
            for k in range(dimensao):
                r1 = rnd.random()
                r2 = rnd.random()

                enxame[i].velocidade[k] = ((w * enxame[i].velocidade[k]) +
                                           (c1 * r1 * (enxame[i].melhor_pos_particula[k] -
                                                       enxame[i].posicao[k])) +
                                           (c2 * r2 * (melhor_pos_enxame[k] -
                                                       enxame[i].posicao[k])))

                if enxame[i].velocidade[k] < minimo:
                    enxame[i].velocidade[k] = minimo
                elif enxame[i].velocidade[k] > maximo:
                    enxame[i].velocidade[k] = maximo

            # atualiza posição
            for k in range(dimensao):
                enxame[i].posicao[k] += enxame[i].velocidade[k]

            # calcula erro da nova posição
            enxame[i].erro = erro(enxame[i].posicao)

            # verifica se encontrou novo melhor pessoal
            if enxame[i].erro < enxame[i].melhor_erro_particula:
                enxame[i].melhor_erro_particula = enxame[i].erro
                enxame[i].melhor_pos_particula = copy.copy(enxame[i].posicao)

            # verifica se encontrou novo melhor global
            if enxame[i].erro < melhor_erro_enxame:
                melhor_erro_enxame = enxame[i].erro
                melhor_pos_enxame = copy.copy(enxame[i].posicao)

        epoca += 1

    return melhor_pos_enxame

# ------------------------------------

print("\nInício da demonstração do PSO em Python\n")

dimensao = 3
print("Objetivo: resolver a função de Rastrigin em " +
      str(dimensao) + " variáveis")
print("A função tem mínimo conhecido = 0.0 em (", end="")
for i in range(dimensao - 1):
    print("0, ", end="")
print("0)")

num_particulas = 50
max_epocas = 100

print("Número de partículas = " + str(num_particulas))
print("Número máximo de épocas = " + str(max_epocas))
print("\nIniciando algoritmo PSO...\n")

melhor_pos = resolver(max_epocas, num_particulas,
                      dimensao, -10.0, 10.0)

print("\nPSO concluído\n")
print("Melhor solução encontrada:")
mostrar_vetor(melhor_pos)
erro_final = erro(melhor_pos)
print("Erro da melhor solução = %.6f" % erro_final)

print("\nFim da demonstração do PSO\n")
