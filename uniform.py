import numpy as np
import matplotlib.pyplot as plt

def uniform_search(f, a, b, N):
    h = (b - a) / (N + 1)
    xs = np.array([a + i*h for i in range(1, N+1)])
    fs = f(xs)
    k = np.argmin(fs)
    left = xs[k-1] if k > 0 else a
    right = xs[k+1] if k < N-1 else b
    print(fs)
    return xs, fs, xs[k], (left, right)

def plot_uniform(f, a, b, xs, fs, interval):
    x = np.linspace(a, b, 400)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linewidth=0.5, alpha=0.7)
    plt.axvline(x=0, color='black', linewidth=0.5, alpha=0.7)
    plt.plot(x, f(x))
    plt.scatter(xs, fs)
    plt.axvline(interval[0], linestyle="--")
    plt.axvline(interval[1], linestyle="--")
    plt.show()

# пример
def f(x):
    return 2*x**2 - 2*x +5/2

a, b = -1, 9
N = 19

xs, fs, xk, interval = uniform_search(f, a, b, N)
print("x* ≈", xk,'+-0.5')
print("f(x)* ≈", f(xk))

print("interval:", f'[{interval[0]}, {interval[1]}]')

plot_uniform(f, a, b, xs, fs, interval)