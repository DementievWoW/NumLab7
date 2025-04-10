import numpy as np
import matplotlib.pyplot as plt
def f(x):
    return x ** 3 + 10 * np.sin(5 * x)

x_nodes = np.array([0, 0.5, 1.2, 1.8, 2.3, 3.0])
y_nodes = f(x_nodes)

def newton_interpolation(x_nodes, y_nodes):
    n = len(x_nodes)
    coef = np.zeros([n, n])
    coef[:, 0] = y_nodes

    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x_nodes[i + j] - x_nodes[i])

    def g(x_val):
        result = coef[0][0]
        for j in range(1, n):
            term = coef[0][j]
            for i in range(j):
                term *= (x_val - x_nodes[i])
            result += term
        return result

    return g

g = newton_interpolation(x_nodes, y_nodes)

x_eval = np.array([0.3, 1.5, 2.7])

results = []
for x in x_eval:
    g_x = g(x)
    f_x = f(x)
    deviation = abs(g_x - f_x)
    results.append((x, g_x, f_x, deviation))

print("x\tg(x)\tf(x)\tОтклонение")
for row in results:
    print(f"{row[0]:.2f}\t{row[1]:.6f}\t{row[2]:.6f}\t{row[3]:.6f}")

x_plot = np.linspace(0, 3, 100)
y_plot_f = f(x_plot)
y_plot_g = [g(x) for x in x_plot]

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot_f, label='f(x) = x^3 + 10*sin(5x)', linewidth=2)
plt.plot(x_plot, y_plot_g, label='Интерполяционная функция g(x)', linestyle='--', linewidth=2)
plt.scatter(x_nodes, y_nodes, color='red', label='Узлы интерполяции')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Интерполяция полиномом Ньютона')
plt.legend()
plt.grid(True)
plt.show()
