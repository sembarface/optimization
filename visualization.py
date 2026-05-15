import numpy as np
import matplotlib.pyplot as plt


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


def plot_halving(f, a0, b0, history):
    x = np.linspace(a0, b0, 400)
    plt.plot(x, f(x))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linewidth=0.5, alpha=0.7)
    plt.axvline(x=0, color='black', linewidth=0.5, alpha=0.7)
    for a, b, y, xc, z in history:
        plt.axvline(a, linestyle="--")
        plt.axvline(b, linestyle="--")
        plt.scatter([y, xc, z], [f(y), f(xc), f(z)])
    plt.show()


def plot_one_dim(f, a, b, history, x_star):
    x = np.linspace(a, b, 400)
    plt.plot(x, f(x))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linewidth=0.5, alpha=0.7)
    plt.axvline(x=0, color='black', linewidth=0.5, alpha=0.7)
    plt.scatter(x_star, f(x_star), color='r')
    for a, b in history:
        plt.axvline(a, linestyle="--")
        plt.axvline(b, linestyle="--")
    plt.show()



def plot_swann(f, result, history):
    if result is None:
        return
    a, b = result
    x = np.linspace(a, b, 400)
    xl = np.linspace(a - (b - a), a, 400)
    xr = np.linspace(b, b + (b - a), 400)

    plt.figure(figsize=(8, 5))
    plt.scatter(history, f(history), s=25)
    plt.plot(xl, f(xl), 'b--', alpha=0.5)
    plt.plot(xr, f(xr), 'b--', alpha=0.5)
    plt.plot(x, f(x), 'b')
    plt.axvline(a, color='r', alpha=0.5)
    plt.axvline(b, color='r', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linewidth=0.5, alpha=0.7)
    plt.axvline(x=0, color='black', linewidth=0.5, alpha=0.7)
    plt.axis('equal')
    plt.show()


def plot_descent(f, history, x_star):
    plt.figure()
    x1 = np.linspace(1, 7, 1000)
    x2 = np.logspace(-7, 0, 100, endpoint=False)
    x = np.concatenate([-x1[::-1], -x2[::-1], [0.0], x2, x1])
    X, Y = np.meshgrid(x, x)
    X += x_star[0]
    Y += x_star[1]
    z = [f(i) for i in history]
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linewidth=0.5, alpha=0.7)
    plt.axvline(x=0, color='black', linewidth=0.5, alpha=0.7)
    plt.contour(X, Y, f(np.array([X, Y])), levels=np.unique(np.sort(z)))
    plt.scatter(history[:, 0], history[:, 1])
    plt.plot(history[:, 0], history[:, 1])
    plt.axis('equal')
    plt.show()


def plot_constraint(f, constraints, history, x_star):
    plt.figure()
    x1 = np.linspace(1, 7, 1000)
    x2 = np.logspace(-7, 0, 100, endpoint=False)
    x = np.concatenate([-x1[::-1], -x2[::-1], [0.0], x2, x1])
    X, Y = np.meshgrid(x, x)
    X += x_star[0]
    Y += x_star[1]
    z = [f(i) for i in history]
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linewidth=0.5, alpha=0.7)
    plt.axvline(x=0, color='black', linewidth=0.5, alpha=0.7)
    plt.contour(X, Y, f(np.array([X, Y])), levels=np.unique(np.sort(z)))
    plt.scatter(history[:, 0], history[:, 1])
    plt.plot(history[:, 0], history[:, 1])
    plt.axis('equal')

    h, g = constraints(np.zeros(2))

    for k in range(len(h)):
        H = np.vectorize(
            lambda x1, x2:
            constraints(np.array([x1, x2]))[0][k]
        )(X, Y)

        plt.contour(X, Y, H, levels=[0], colors='red')

    for k in range(len(g)):
        G = np.vectorize(
            lambda x1, x2:
            constraints(np.array([x1, x2]))[1][k]
        )(X, Y)

        plt.contour(X, Y, G, levels=[0], colors='blue')
        plt.contourf(X, Y, G, levels=[G.min(), 0], alpha=0.2, colors='blue')

    plt.show()


def plot_surface_descent(f, history, x_star):
    z_hist = np.array([f(x) for x in history])

    r = 2.5 * max(np.abs(history - x_star).max(), 1.0)
    x = np.linspace(-r, r, 80)
    X, Y = np.meshgrid(x_star[0] + x, x_star[1] + x)

    Z = np.vectorize(lambda a, b: f(np.array([a, b])))(X, Y)

    dz = max(z_hist.max() - z_hist.min(), 1.0)
    h = 0.02 * dz

    step = max(len(history) // 8, 1)
    levels = np.unique(np.round(z_hist[::step], 6))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, color='blue', alpha=0.3, linewidth=0, shade=True)
    ax.contour(X, Y, Z + h, levels=levels, colors='g', linewidths=1.2)

    ax.plot(history[:, 0], history[:, 1], z_hist + h, color='crimson', linewidth=2.5)
    ax.scatter(history[:, 0], history[:, 1], z_hist + h, color='black', s=28, depthshade=False)

    ax.set_zlim(z_hist.min() - 0.1 * dz, z_hist.max() + 0.2 * dz)
    ax.set_box_aspect((1, 1, 1.1))

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x)')
    ax.view_init(elev=28, azim=-55)

    plt.show()
