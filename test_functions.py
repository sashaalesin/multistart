from functools import reduce

from numpy import cos, exp, pi, sqrt


def check_dim(dim, min=1, max=100):
    assert (
        type(dim) == int and dim >= min and dim <= max
    ), f'Размерность должна быть в пределах [{min}, {max}] для этой функции'


def check_bounds(dim, bounds):
    assert (
        len(bounds) == dim and reduce(
            lambda acc, bound: acc and len(bound) == 2,
            bounds,
            True
        )
    ), 'Границы функции заданы неверно'


class Sphere:
    def __init__(self, dim=2, bounds=None):
        check_dim(dim)
        self.dim = dim

        if bounds:
            check_bounds(bounds)
            self.bounds = bounds
        else:
            self.bounds = [[-3, 3] for _ in range(self.dim)]

    def __call__(self, x):
        return sum(coord ** 2 for coord in x)

    def __repr__(self):
        return 'Sphere'


class Rosenbrock:
    def __init__(self, dim=2, bounds=None):
        check_dim(dim)
        self.dim = dim

        if bounds:
            check_bounds(bounds)
            self.bounds = bounds
        else:
            if self.dim == 1:
                self.bounds = [[-2, 2]]
            if self.dim == 2:
                self.bounds = [[-2, 2], [-1, 3]]
            else:
                self.bounds = [[-3, 3] for _ in range(self.dim)]

    def __call__(self, x):
        if self.dim == 1:
            return (1 - x[0]) ** 2
        s = 0
        for i in range(1, self.dim):
            s += 100 * (x[i] - x[i - 1] ** 2) ** 2 + (1 - x[i - 1]) ** 2
        return s

    def __repr__(self):
        return 'Rosenbrock'


class Rastrigin:
    def __init__(self, dim=2, bounds=None):
        check_dim(dim)
        self.dim = dim

        if bounds:
            check_bounds(bounds)
            self.bounds = bounds
        else:
            self.bounds = [[-5.12, 5.12] for _ in range(self.dim)]

    def __call__(self, x):
        A = 10
        s = A * self.dim
        for i in range(self.dim):
            s += x[i] ** 2 - A * cos(2 * pi * x[i])
        return s

    def __repr__(self):
        return 'Rastrigin'


class Himmelblau:
    def __init__(self, dim=2, bounds=None):
        check_dim(dim, 2, 2)
        self.dim = dim

        if bounds:
            check_bounds(bounds)
            self.bounds = bounds
        else:
            self.bounds = [[-4.5, 4.5], [-4.5, 4.5]]

    def __call__(self, x):
        x1, x2 = x
        return (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2

    def __repr__(self):
        return 'Himmelblau'


class Easom:
    def __init__(self, dim=2, bounds=None):
        check_dim(dim, 2, 2)
        self.dim = dim

        if bounds:
            check_bounds(bounds)
            self.bounds = bounds
        else:
            self.bounds = [[-10, 10], [-10, 10]]

    def __call__(self, x):
        x1, x2 = x
        return -cos(x1) * cos(x2) * exp(-((x1 - pi) ** 2 + (x2 - pi) ** 2))

    def __repr__(self):
        return 'Easom'


class Ackley:
    def __init__(self, dim=2, bounds=None):
        check_dim(dim, 2, 2)
        self.dim = dim

        if bounds:
            check_bounds(bounds)
            self.bounds = bounds
        else:
            self.bounds = [[-5, 5], [-5, 5]]

    def __call__(self, x):
        x1, x2 = x
        return -20 * exp(
            -0.2 * sqrt(0.5 * (x1 ** 2 + x2 ** 2))
        ) - exp(
            0.5 * (cos(2 * pi * x1) + cos(2 * pi * x2))
        ) + exp(1) + 20

    def __repr__(self):
        return 'Ackley'
