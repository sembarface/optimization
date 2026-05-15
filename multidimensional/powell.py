import numpy as np

from one_dimensional.fibonacci import fib_search
from one_dimensional.swann import swann
from visualization import plot_descent
import time

def powell(f, x0, eps, M):
    n = len(x0)
    history = [x0.copy()]
    xk = x0.copy()
    x_star = x0.copy()

    d = np.eye(n)
    d = np.vstack([d[-1], d])
    yi = xk.copy()
    y0 = yi.copy()
    for k in range(M+1):
        for i in range(n+1):
            phi = lambda t: f(yi + t * d[i])

            res = swann(phi, 0, t=np.sqrt(eps))

            if res is None:
                continue
            interval, _ = res

            t, _ = fib_search(phi, *interval, eps)
            yi = yi + t * d[i]
            if i == 0:
                y1 = yi.copy()
            if (i == n-1) and (np.linalg.norm(yi - y0) < eps):
                x_star = yi
                history.append(x_star)
                return x_star, k, np.array(history)
            
            if (i == n) and (np.linalg.norm(yi - y1) < eps):
                x_star = yi
                history.append(x_star)
                return x_star, k, np.array(history)
        
        x_next = yi
        if np.linalg.norm(x_next - xk) < eps:
            x_star = x_next
            history.append(x_star)
            return x_star, k, np.array(history)

        d_ = np.delete(d, 1,axis=0)
        d_[0] = yi - y1
        d_ = np.vstack([d_, yi - y1])
        if np.linalg.matrix_rank(d_[1:]) == n:
            d = d_
            y0 = x_next
        else:
            y0 = x_next


        xk = x_next
        history.append(xk)

    x_star = xk
    return x_star, k, np.array(history)


if __name__ == '__main__':

    def f(x):
        return 8*x[0]**2 + x[1]**2 - x[0] * x[1] + x[0]
    
    x0 = np.array([2, 2])
    eps = 1e-5
    M = 1000

    
    start = time.perf_counter()
    x_star, k, history = powell(f, x0, eps, M)
    end = time.perf_counter()

    print("x* ≈", x_star)
    print("f(x*) ≈", f(x_star))
    print('Итераций:', k)
    
    print("Время:", end - start)
    plot_descent(f,history, x_star)
    
