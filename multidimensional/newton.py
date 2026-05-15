import numpy as np

from visualization import plot_descent
import time


def newton(f, grad, hessian, x0, eps, M):
    history = [x0.copy()]
    xk = x0.copy()
    x_star = xk
    x_pred = xk.copy()
    fk = f(xk)
    f_pred = fk
    for k in range(M+1):
        g = grad(xk)
        print(g,'   ', np.linalg.norm(g))
        if np.linalg.norm(g) < eps:
            x_star = xk
            print("Маленький градиент")
            break
        
        h = hessian(xk)
        # h_inv = np.linalg.inv(h)
        if np.all(np.linalg.eigvalsh(h) > 0):
            d = np.linalg.solve(h, -g)
            # d = - h_inv @ g
            x_next = xk + d
        else:
            d = -g
            t = 1
            while f(xk - t*g) > fk - 1e-4*t*np.dot(g,g):
                t *= 0.5
            x_next = xk + t * d

        f_next = f(x_next)
        
        if (np.linalg.norm(x_next - xk) < eps and abs(f_next - fk) < eps and 
            np.linalg.norm(xk - x_pred) < eps and abs(fk - f_pred) < eps):
            xk = x_next
            history.append(xk)
            print(f"Малая разница x и f")
            print(f'Норма разницы f = {abs(f_next - fk)}')
            break

        x_pred = xk
        f_pred = fk
        xk = x_next
        fk = f_next
        history.append(xk)

    x_star = xk
    return x_star, k, np.array(history)

if __name__ == '__main__':

    def f(x):
        return (x[0] + 2*x[1] -5)**4 +(x[1]-2)**2 + 3 + (x[0] + x[1] + x[2] -5) ** 2
    
    x0 = np.array([100, 50, -20])
    eps = 1e-5
    M = 1000

    # def grad(x):
    #     return np.array([16 * x[0] - x[1] + 1, 2 * x[1] - x[0]])

    def grad(x, eps=1e-5):
        return np.array([(f(x+d)-f(x-d))/(2*eps) for d in np.eye(len(x))*eps])
    
    # def hessian(x):
    #     return np.array([[16, -1],
    #                     [-1, 2 ]])

    def hessian(x, eps=1e-5):
        n = len(x)
        H = np.zeros((n, n))
        E = np.eye(n) * eps
        for i in range(n):
            for j in range(i, n):
                ei, ej = E[i], E[j]
                val = (f(x + ei + ej) - f(x + ei - ej)
                    - f(x - ei + ej) + f(x - ei - ej)) / (4 * eps**2)
                H[i, j] = val
                H[j, i] = val
        return H

    start = time.perf_counter()
    x_star, k, history = newton(f, grad, hessian, x0, eps, M)
    end = time.perf_counter()

    print("x* ≈", x_star)
    print("f(x*) ≈", f(x_star))
    print('Итераций:', k)
    
    print("Время:", end - start)
    # plot_descent(f,history, x_star)
