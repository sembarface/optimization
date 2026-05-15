import numpy as np

from visualization import plot_one_dim

def gold(f, a, b, eps):
    history = []
    phi = (np.sqrt(5) - 1) / 2  
    
    y = b - phi * (b - a)
    z = a + phi * (b - a)
    f_y = f(y)
    f_z = f(z)

    while (b - a) > 2 * eps:
        print(np.array([a, y, z, b]))

        if f_y <= f_z:
            b = z
            z = y
            f_z = f_y
            y = b - phi * (b - a)
            f_y = f(y)
        else:
            a = y
            y = z
            f_y = f_z
            z = a + phi * (b - a)
            f_z = f(z)

        history.append([a, b])

    return (a + b) / 2, np.array(history)

if __name__ == '__main__':
    def f(x):
        return x**2 * np.log(x)

    a, b = 0.1, 100
    eps = 0.005

    x_star, history = gold(f, a, b, eps)
    print("x* ≈", x_star,'+-',eps)
    print("f(x)* ≈", f(x_star))

    plot_one_dim(f, a, b, history,x_star)
        
