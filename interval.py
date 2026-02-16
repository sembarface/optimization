import numpy as np
import matplotlib.pyplot as plt

def interval_halving(f, a, b, eps):
    history = []
    while (b - a) > eps:
        L = b - a
        xc = (a + b) / 2
        y = a + L/4
        z = b - L/4
        print(np.array([a, y, xc, z, b]))
        if f(y) < f(xc):
            b = xc
        elif f(z) < f(xc):
            a = xc
        else:
            a, b = y, z


        history.append((a, b, y, xc, z))
    return (a + b)/2, history

def plot_halving(f, a0, b0, history):
    x = np.linspace(a0, b0, 400)
    plt.plot(x, f(x))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linewidth=0.5, alpha=0.7)
    plt.axvline(x=0, color='black', linewidth=0.5, alpha=0.7)
    for a, b, y, xc, z in history:
        plt.axvline(a, linestyle="--")
        plt.axvline(b, linestyle="--")
        plt.scatter([y, xc, z], [f(y), f(xc), f(z)])
    plt.show()

# пример
def f(x):
    return 2*x**2 - 2*x +5/2

a, b = -1, 9
eps = 0.5

x_star, history = interval_halving(f, a, b, eps)
print("x* ≈", x_star,'+-',eps)
print("f(x)* ≈", f(x_star))

plot_halving(f, a, b, history)