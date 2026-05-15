import numpy as np

from visualization import plot_swann

def swann(f, x0, t=1e-2, M=100):
    history = [x0]

    f_l = f(x0 - t)
    f_m = f(x0)
    f_r = f(x0 + t)

    if f_l >= f_m <= f_r:
        return [x0 - t, x0 + t], np.array(history)

    if f_l <= f_m >= f_r:
        return None

    if f_l >= f_m >= f_r:
        delta = t
        a0 = x0
        xk = x0 + t
    else:
        delta = -t
        b0 = x0
        xk = x0 - t

    history.append(xk)

    k = 1
    f_k = f(xk)

    while k <= M:
        x_next = xk + 2**k * delta
        f_next = f(x_next)
        history.append(x_next)

        if f_next < f_k:
            if delta > 0:
                a0 = xk
            else:
                b0 = xk

            xk = x_next
            f_k = f_next
            k += 1
        else:
            if delta > 0:
                b0 = x_next
            else:
                a0 = x_next

            return [a0, b0], np.array(history)

    return None

if __name__ == '__main__':
    def f(x):
        return -1.5*np.exp(-x**2)

    result, history = swann(f, -1)
    print(f'Интервал: {result}')
    print(f'Точки: {history}')
    print(f'Итераций: {len(history) - 1}')

    plot_swann(f, result, history)
