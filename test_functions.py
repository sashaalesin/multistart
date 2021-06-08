from numpy import cos, pi


def rosenbrock(x):
    n = len(x)
    if n == 1:
        return (1 - x[0]) ** 2
    s = 0
    for i in range(1, n):
        s += 100 * (x[i] - x[i - 1] ** 2) ** 2 + (1 - x[i - 1]) ** 2
    return s


def rastrigin(x):
    n = len(x)
    A = 10
    s = A * n
    for i in range(n):
        s += x[i] ** 2 - A * cos(2 * pi * x[i])
    return s


def himmelblau(x):
    x1, x2 = x
    return (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2


f = {
    'rosenbrock': rosenbrock,
    'rastrigin': rastrigin,
    'himmelblau': himmelblau,
}

bounds2d = {
    'rastrigin': [[-5.12, 5.12]],
    'rosenbrock': [[-2, 2]],
}

bounds3d = {
    'himmelblau': [[-4.5, 4.5], [-4.5, 4.5]],
    'rastrigin': [[-5.12, 5.12], [-5.12, 5.12]],
    'rosenbrock': [[-2, 2], [-1, 3]],
}

bounds4d = {
    'rastrigin': [[-5.12, 5.12], [-5.12, 5.12], [-5.12, 5.12]],
    'rosenbrock': [[-3, 3], [-3, 3], [-3, 3]],
}
