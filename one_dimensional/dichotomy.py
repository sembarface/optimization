import numpy as np

from visualization import plot_one_dim

def dichotomy(f, a, b, eps, delta):
    history = []
    
    while (b - a) > 2*eps:    
        y = (a+b-delta)/2
        z = (a+b+delta)/2
        print(np.array([a,y,z,b]))
        if f(y) <= f(z):
            b = z
        else:
            a = y
        
        history.append([a,b])
    
    return (a+b)/2, np.array(history)

if __name__ == '__main__':
    def f(x):
        return 2*x**2 - 2*x +5/2

    a, b = -1, 9
    eps = 0.005
    delta = eps*0.1

    x_star, history = dichotomy(f, a, b, eps, delta)
    print("x* ≈", x_star,'+-',eps)
    print("f(x)* ≈", f(x_star))

    plot_one_dim(f, a, b, history,x_star)
        
