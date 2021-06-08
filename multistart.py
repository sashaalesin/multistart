import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans

from plotter import plot_result
from printer import print_result
from sequences import s
from test_functions import bounds2d, bounds3d, bounds4d, f

MAX_ITERATION_COUNT = 10


def min_point(cluster, func_name):
    # Находит точки из кластера,
    # в которых функция принимает минимальное значение
    funcs_in_points = list(map(
        lambda point: f[func_name](point),
        cluster
    ))
    idx_of_min = [0]
    min_value = funcs_in_points[0]
    for idx in range(1, len(funcs_in_points)):
        value = funcs_in_points[idx]
        if value < min_value:
            min_value = value
            idx_of_min = [idx]
        elif value == min_value:
            idx_of_min.append(idx)
    return [cluster[idx] for idx in idx_of_min]


def get_clusters(points, func_name, n_clusters=2):
    # Разбивает точки на кластеры
    km = KMeans(n_clusters=n_clusters, max_iter=100)
    km.fit(points)

    clusters = [[] for _ in range(n_clusters)]
    for i in range(len(points)):
        clusters[km.labels_[i]].append(np.array(points[i]))

    min_points = []
    for cluster in clusters:
        min_points.append(min_point(cluster, func_name))

    return {
        'centroids': km.cluster_centers_,
        'clusters': list(map(lambda cluster: np.array(cluster), clusters)),
        'minPoints': min_points,
    }


def merge_lists_and_get_unique(lst):
    # Системная функция для слияния массивов и получения уникальных точек
    return np.unique([p for cluster in lst for p in cluster], axis=0)


def multistart(**kwargs):
    # Метод мультистарт
    # Получаем входные данные
    func_name = kwargs['func_name']
    sequence_name = kwargs['sequence_name']
    dim = kwargs.get('dim') or 2
    n = kwargs.get('n') or 32
    bounds = kwargs.get('bounds') or [[0, 1] for _ in range(dim)]
    n_clusters = kwargs.get('n_clusters') or 8
    # Генерируем точки
    points = s[sequence_name](dim=dim, n=n, bounds=bounds)

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
                    f[func_name],
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
            func_name,
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

        result['answer'] = [[np.round(x, 5) for x in p] for p in points]

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
            func_name,
            bounds,
            points=kwargs.get('plot_points'),
            surface=kwargs.get('plot_surface'),
        )

    return result


# Имя функции (доступные в файле test_functions.py)
func_name = 'rastrigin'
# Имя последовательности (доступные в файле sequences.py)
sequence_name = 'halton'

result = multistart(
    func_name=func_name,
    sequence_name=sequence_name,
    # Размерность
    dim=1,
    # Количество точек
    n=32,
    # Область определения (поиска) (описаны в test_functions.py)
    bounds=bounds2d[func_name],
    # Начальное число кластеров
    n_clusters=10,
    # Строить ли точки (Доступно только для 2d и 3d областей (3d и 4d функций))
    plot_points=True,
    # Строить ли функции (Доступно только для 3d функций)
    plot_surface=True,
    # Выводить ли ответ в консоль
    print=False,
)
