import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans

from plotter import plot_result
from printer import print_result
from sequences import seq
from test_functions import (Ackley, Easom, Himmelblau, Rastrigin, Rosenbrock,
                            Sphere)

MAX_ITERATION_COUNT = 10
ROUND = 5


def get_min_points_from_cluster(cluster, func):
    # Находит точки из кластера,
    # в которых функция принимает минимальное значение
    funcs_in_points = np.array(list(map(
        lambda point: func(point),
        cluster
    )))
    min_value = funcs_in_points.min()
    return [
        cluster[i] for i in range(len(cluster))
        if funcs_in_points[i] == min_value
    ]


def get_clusters(points, func, n_clusters=2):
    # Разбивает точки на кластеры
    km = KMeans(n_clusters=n_clusters, max_iter=100)
    km.fit(points)

    clusters = [[] for _ in range(n_clusters)]
    for i in range(len(points)):
        clusters[km.labels_[i]].append(np.array(points[i]))

    min_points = [
        get_min_points_from_cluster(cluster, func)
        for cluster in clusters
    ]

    return {
        'clusters': list(map(lambda cluster: np.array(cluster), clusters)),
        'minPoints': min_points,
    }


def merge_lists_and_get_unique(lst):
    # Системная функция для слияния массивов и получения уникальных точек
    return np.unique(
        [np.round(p, ROUND) for cluster in lst for p in cluster],
        axis=0
    )


def multistart(**kwargs):
    # Метод мультистарт
    # Получаем входные данные
    func = kwargs['func']
    sequence_name = kwargs['sequence_name']
    dim = kwargs.get('dim') or 2
    n = kwargs.get('n') or 32
    bounds = kwargs.get('bounds') or func.bounds
    n_clusters = kwargs.get('n_clusters') or 8
    # Генерируем точки
    points = seq(sequence_name=sequence_name, dim=dim, n=n, bounds=bounds)

    result = {
        'dim': dim,
        'n': n,
        'sequenceName': sequence_name,
        0: {'points': points}
    }

    iteration_counter = 0
    number_of_clusters = n_clusters
    while len(points) > 1 and iteration_counter < MAX_ITERATION_COUNT:
        iteration_counter += 1

        result[iteration_counter] = {}

        local_minimums = np.unique(
            [
                minimize(
                    func,
                    x0,
                    method='Nelder-Mead',
                    tol=0.01,
                ).x for x0 in points
            ],
            axis=0
        )
        result[iteration_counter]['localMinimums'] = local_minimums

        clustered_local_minimums = get_clusters(
            local_minimums,
            func,
            n_clusters=number_of_clusters
        )
        result[iteration_counter][
            'clusteredLocalMinimums'
        ] = clustered_local_minimums

        new_points = merge_lists_and_get_unique(
            clustered_local_minimums['minPoints']
        )

        len_points = len(points)
        len_min_points = len(new_points)

        points = new_points

        number_of_clusters = int(
            np.round(number_of_clusters * (len_min_points / len_points))
        ) + 1

        result['answer'] = [[np.round(x, ROUND) for x in p] for p in points]

        if len_points == len_min_points and number_of_clusters != 1:
            number_of_clusters = 1
            continue

        if len_points == len_min_points and number_of_clusters == 1:
            break

    result['numberOfIteration'] = iteration_counter

    if kwargs.get('print'):
        print_result(result)

    if kwargs.get('plot_points') or kwargs.get('plot_surface'):
        plot_result(
            result,
            func,
            bounds,
            points=kwargs.get('plot_points'),
            surface=kwargs.get('plot_surface'),
        )

    return result


# Размерность области поиска
dim = 1
# Имя функции (доступные в файле test_functions.py)
func = Rastrigin(dim)
# Имя последовательности (доступные в файле sequences.py)
sequence_name = 'halton'

result = multistart(
    func=func,
    sequence_name=sequence_name,
    dim=dim,
    # Количество точек
    n=32,
    # Область определения (поиска) (описаны в test_functions.py)
    bounds=None,
    # Начальное число кластеров
    n_clusters=10,
    # Строить ли точки (Доступно только для 2d и 3d областей (3d и 4d функций))
    plot_points=True,
    # Строить ли функции (Доступно только для 2d и 3d функций)
    plot_surface=True,
    # Выводить ли ответ в консоль
    print=False,
)
