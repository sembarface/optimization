import numpy as np

from visualization import plot_one_dim

def fibonacci_numbers_up_to(limit):
    fib = [1, 1]
    while fib[-1] < limit:
        fib.append(fib[-1] + fib[-2])
    return fib

def fib_search(f, a, b, eps):
    history = []
    
    L0 = (b - a) / (2 * eps)
    fib = fibonacci_numbers_up_to(L0)
    n = len(fib) - 1

    z = a + fib[n - 1] / fib[n] * (b - a)
    y = a + b - z
    f_y = f(y)
    f_z = f(z)

    for k in range(1, n - 1):
        # print(np.array([a, y, z, b]))

        if f_y <= f_z:
            b = z
            z = y
            f_z = f_y
            y = a + b - z
            f_y = f(y)
        else:
            a = y
            y = z
            f_y = f_z
            z = a + b - y
            f_z = f(z)

        history.append([a, b])
    # print(y,z)
    return (a + b) / 2, np.array(history)

if __name__ == '__main__':
    def f(x):
        return 2*x**2 - 2*x + 5/2

    a, b = -1, 9
    eps = 0.005



    x_star, history = fib_search(f, a, b, eps)
    print("x* ≈", x_star, "+-", eps)
    print("f(x*) ≈", f(x_star))

    plot_one_dim(f, a, b, history, x_star)
