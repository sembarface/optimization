import numpy as np

from multidimensional.powell import powell
from visualization import plot_constraint
import time


def lagrange(f, constraints, x0, r0, c, lam0, mu0, eps, M):
    def P(x, muk, rk):
        h, g = constraints(x)
        return (rk/2 * np.sum(h ** 2) +
                 1/(2*rk) * np.sum(np.maximum(0,muk + rk *g)**2 - muk**2))

    def F(x, lamk, muk, rk):
        h, g = constraints(x)
        return f(x) + np.sum(lamk * h)  + P(x, muk, rk)
    
    rk = r0
    xk = x0.copy()
    lamk = lam0.copy()
    muk = mu0.copy()
    history = [xk.copy()]

    for k in range(M+1):
        Frk = lambda x: F(x,lamk=lamk, muk=muk, rk=rk)
        xrk, _, _  = powell(Frk, xk, eps, M)
        
        p = P(xrk, muk, rk)

        if p<= eps:
            xk = xrk
            break
        else:
            h, g = constraints(xrk)
            lamk += rk * h
            muk = np.maximum(0,muk + rk *g)
            rk *= c
            xk = xrk
        history.append(xk.copy())
    
    return xk, k, np.array(history)


if __name__ == '__main__':

    def f(x):
        return 2*x[0]**2 + 7*x[1]**2 -x[0]*x[1] + x[0]

    def constraints(x):
        h = np.array([x[0] + 4*x[1] - 3])
        g = np.array([])
        return h, g

    # def f(x):
    #     return 8*x[0]**2 + x[1]**2 - x[0] * x[1] + x[0]
    
    # def constraints(x):
    #     h = np.array([2*x[0] + x[1] - 3])
    #     g = np.array([])
    #     return h, g
    
    x0 = np.array([2, 2])
    eps = 1e-5
    M = 1000
    r0 = 1
    c = 10
    lam0 = np.array([1.0])
    mu0 = np.array([])


    start = time.perf_counter()
    x_star, k, history = lagrange(f, constraints, x0, r0, c, lam0, mu0, eps, M)
    end = time.perf_counter()

    print("x* ≈", x_star)
    print("f(x*) ≈", f(x_star))
    print('Итераций:', k)
    
    print("Время:", end - start)

    plot_constraint(f, constraints, history, x_star)
