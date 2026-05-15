import numpy as np

from visualization import plot_uniform

def uniform_search(f, a, b, N):
    h = (b - a) / (N + 1)
    xs = np.array([a + i*h for i in range(1, N+1)])
    fs = f(xs)
    k = np.argmin(fs)
    left = xs[k-1] if k > 0 else a
    right = xs[k+1] if k < N-1 else b
    print(fs)
    return xs, fs, xs[k], np.array([left, right])

if __name__ == '__main__':
    
    def f(x):
        return 2*x**2 - 2*x +5/2

    a, b = -1, 9
    N = 19

    xs, fs, xk, interval = uniform_search(f, a, b, N)
    print("x* ≈", xk,'+-0.5')
    print("f(x)* ≈", f(xk))

    print("interval:", f'[{interval[0]}, {interval[1]}]')

    plot_uniform(f, a, b, xs, fs, interval)
