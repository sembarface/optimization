import numpy as np

from multidimensional.powell import powell
from visualization import plot_constraint
import time


def barrier(f, constraints, x0, r0, c, eps, M):
    def P(x, rk):
        h, g = constraints(x)
        return -rk * np.sum(1/g)

    def F(x,rk):
        return f(x) + P(x, rk)
    rk = r0
    xk = x0.copy()
    history = [xk.copy()]
    for k in range(M+1):
        Frk = lambda x: F(x,rk=rk)
        xrk, _, _  = powell(Frk, xk, eps, M)
        p = P(xrk, rk)
        if p<= eps:
            xk = xrk
            break
        else:
            rk /= c
            xk = xrk
        history.append(xk.copy())
    
    return xk, k, np.array(history)



if __name__ == '__main__':

    # def f(x):
    #     return 2*x[0]**2 + 7*x[1]**2 -x[0]*x[1] + x[0]

    # def constraints(x):
    #     h = np.array([])
    #     g = np.array([2*x[0] + x[1] + 1])
    #     return h, g

    def f(x):
        return 8*x[0]**2 + x[1]**2 - x[0] * x[1] + x[0]
    
    def constraints(x):
        h = np.array([])
        g = np.array([2*x[0] + x[1] + 1])
        return h, g
    
    x0 = np.array([-5, -5])
    eps = 1e-5
    M = 1000
    r0 = 0.1
    c = 10



    start = time.perf_counter()
    x_star, k, history = barrier(f, constraints, x0, r0, c, eps, M)
    end = time.perf_counter()

    print("x* ≈", x_star)
    print("f(x*) ≈", f(x_star))
    print('Итераций:', k)
    
    print("Время:", end - start)

    plot_constraint(f, constraints, history, x_star)
