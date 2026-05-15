import numpy as np

from visualization import plot_halving

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
    return (a + b)/2, np.array(history)

if __name__ == '__main__':
    def f(x):
        return 2*x**2 - 2*x +5/2

    a, b = -1, 9
    eps = 0.5

    x_star, history = interval_halving(f, a, b, eps)
    print("x* ≈", x_star,'+-',eps)
    print("f(x)* ≈", f(x_star))

    plot_halving(f, a, b, history)
