import matplotlib.pyplot as plt
import numpy as np

# Definição das restrições
x1 = np.linspace(0, 20, 200)
x2_1 = 6 - x1           # reta x1 + x2 = 6
x2_2 = (18 - x1) / 2    # reta x1 + 2x2 = 18

# Plotagem
plt.figure(figsize=(7,7))
plt.plot(x1, x2_1, label=r'$x_1 + x_2 = 6$')
plt.plot(x1, x2_2, label=r'$x_1 + 2x_2 = 18$')

# Região factível (sombreada)
x_fill = [0, 0, 18, 6]
y_fill = [6, 9, 0, 0]
plt.fill(x_fill, y_fill, color='gray', alpha=0.5)

# Pontos vértices
vertices = [(0,6), (0,9), (6,0), (18,0)]
for (x,y) in vertices:
    plt.plot(x,y,'ro')
    plt.text(x+0.3, y+0.3, f"({x},{y})")

# Ponto ótimo
plt.plot(6,0,'go', markersize=10, label='Ótimo (6,0)')

plt.xlim(0,20)
plt.ylim(0,12)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend()
plt.grid(True)
plt.title("Região Factível - Exemplo 1.1")
plt.show()
