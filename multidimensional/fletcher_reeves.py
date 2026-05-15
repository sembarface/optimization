import numpy as np

from one_dimensional.fibonacci import fib_search
from one_dimensional.swann import swann
from visualization import plot_descent
import time

def fletcher_reeves(f, grad, x0, eps, M):
    n = len(x0)
    history = [x0.copy()]
    xk = x0.copy()
    x_star = xk
    x_pred = xk.copy()
    fk = f(xk)
    f_pred = fk
    d_pred = np.zeros(len(xk))
    g_pred = grad(x_pred)
    for k in range(M+1):
        gk = grad(xk)
        if np.linalg.norm(gk) < eps:
            x_star = xk
            break

        # betta = (gk @ (gk - g_pred)) / (g_pred @ g_pred) if (k % n!=0) else 0
        betta = (gk @ gk) / (g_pred @ g_pred)
        dk = - gk + betta * d_pred

        phi = lambda t: f(xk + t * dk)
        interval, _ = swann(phi,0)
        t, _ = fib_search(phi, *interval, eps)
        # t = 1
        # while f(xk - t*dk) > fk - 1e-4*t*np.dot(g,dk):
        #     t *= 0.5

        x_next = xk + t * dk
        f_next = f(x_next)
        
        if (np.linalg.norm(x_next - xk) < eps and abs(f_next - fk) < eps and 
            np.linalg.norm(xk - x_pred) < eps and abs(fk - f_pred) < eps):
            xk = x_next
            history.append(xk)
            break
        g_pred = gk
        d_pred = dk
        x_pred = xk
        f_pred = fk
        xk = x_next
        fk = f_next
        history.append(xk)

    x_star = xk
    return x_star, k, np.array(history)


if __name__ == '__main__':

    def f(x):
        return 8*x[0]**2 + x[1]**2 - x[0] * x[1] + x[0]
    
    x0 = np.array([2, 2])
    eps = 1e-5
    M = 100

    def grad(x):
        return np.array([16 * x[0] - x[1] + 1, 2 * x[1] - x[0]])

    # def grad(x):
    #     return np.array([(f(x+d)-f(x-d))/(2*eps) for d in np.eye(len(x))*eps])
    
    start = time.perf_counter()
    x_star, k, history = fletcher_reeves(f, grad, x0, eps, M)
    end = time.perf_counter()

    print("x* ≈", x_star)
    print("f(x*) ≈", f(x_star))
    print('Итераций:', k)
    
    print("Время:", end - start)
    plot_descent(f, history, x_star)
