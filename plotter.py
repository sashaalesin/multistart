import matplotlib.pyplot as plt
import numpy as np

from test_functions import f

COLORS = [
    'darkblue',
    'darkcyan',
    'darkgoldenrod',
    'darkgray',
    'darkgreen',
    'darkkhaki',
    'darkmagenta',
    'darkolivegreen',
    'darkorange',
    'darkorchid',
    'darkred',
    'darksalmon',
    'darkseagreen',
    'darkslateblue',
    'darkslategray',
    'darkturquoise',
    'darkviolet',
    'deeppink',
    'deepskyblue',
    'dimgray',
    'dodgerblue',
    'magenta',
    'maroon',
    'mediumaquamarine',
    'mediumblue',
    'mediumorchid',
    'mediumpurple',
    'mediumseagreen',
    'mediumslateblue',
    'mediumspringgreen',
    'mediumturquoise',
    'mediumvioletred',
    'midnightblue',
    'mintcream',
    'mistyrose',
    'moccasin',
    'navajowhite',
    'navy',
    'oldlace',
    'olive',
    'olivedrab',
    'orange',
    'orangered',
    'orchid',
    'palegoldenrod',
    'palegreen',
    'paleturquoise',
    'palevioletred',
    'papayawhip',
    'peachpuff',
    'peru',
    'pink',
    'plum',
    'powderblue',
    'purple',
    'red',
    'rosybrown',
    'royalblue',
    'saddlebrown',
    'salmon',
    'sandybrown',
    'seagreen',
    'seashell',
    'springgreen',
    'steelblue',
    'tan',
    'teal',
    'thistle',
    'tomato',
    'turquoise',
    'violet',
    'wheat',
    'yellow',
    'yellowgreen',
    'aqua',
    'aquamarine',
    'azure',
    'beige',
    'bisque',
    'black',
    'blanchedalmond',
    'blue',
    'blueviolet',
    'brown',
    'burlywood',
    'cadetblue',
    'chartreuse',
    'chocolate',
    'coral',
    'cornflowerblue',
    'cornsilk',
    'crimson',
    'cyan',
]


def plot_surface(name, bounds, ax, dim):
    x = np.arange(*bounds[0], 0.01)
    if dim == 2:
        y = np.arange(*bounds[1], 0.01)
        xgrid, ygrid = np.meshgrid(x, y)
        xy = np.stack([xgrid, ygrid])

        ax.view_init(45, -45)
        ax.plot_surface(xgrid, ygrid, f[name](xy), cmap='terrain')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel(f'{name}(x, y)')
    elif dim == 1:
        y = np.array([f[name]([x_coord]) for x_coord in x])
        ax.plot(x, y)
        ax.set_xlabel('x')
        ax.set_ylabel(f'{name}(x)')


def plot_points(result, bounds, axs):
    dim = result['dim']
    n = result['n']
    number_of_iteration = result['numberOfIteration']

    coords = [result[0]['points'][:, i] for i in range(dim)]
    if dim == 1:
        coords.append([0] * n)

    axs[0].scatter(*coords)
    axs[0].set_title(f"Исходных точек: {n}")
    axs[0].set_ylabel(result['sequenceName'])

    number_of_axs = 1

    for i in range(1, number_of_iteration + 1):
        min_points = result[i]['clusteredLocalMinimums']['minPoints']

        clusters = result[i]['clusteredLocalMinimums']['clusters']
        number_of_clusters = len(clusters)
        number_of_min_points = 0

        for j in range(number_of_clusters):
            cluster = clusters[j]
            min_points_of_this_cluster = min_points[j]
            number_of_min_points += len(min_points_of_this_cluster)

            coords = [cluster[:, i] for i in range(dim)]
            if dim == 1:
                coords.append([0] * len(cluster))

            axs[i].scatter(
                *coords,
                c=COLORS[j],
            )

            for min_point in min_points_of_this_cluster:
                coords = [min_point[i] for i in range(dim)]
                if dim == 1:
                    coords.append([0] * len(min_point))

                axs[i].scatter(
                    *coords,
                    marker='*',
                    c=COLORS[j],
                    s=200,
                )

        axs[i].set_xlim(bounds[0])
        if dim == 2:
            axs[i].set_ylim(bounds[1])
        axs[i].set_title(
            f"{i}. {len(result[i]['localMinimums'])} лок.мин. => {number_of_clusters} кл-ов => {number_of_min_points} точек"
        )
        # axs[i].set_aspect('equal')

        number_of_axs += 1

    axs[-1].set_xlabel(f"Ответ: {result['answer']}")


def plot_result(result, name, bounds, points=False, surface=False):
    if points or surface:
        dim = result['dim']
        number_of_iteration = result['numberOfIteration']

        nrows = points * int(
            np.ceil((number_of_iteration + 1) / 3)
        )
        if dim < 3:
            nrows += surface

        axes = []

        if points:
            if dim <= 3:
                for i in range(number_of_iteration + 1):
                    axes.append(
                        plt.subplot2grid(
                            shape=(nrows, 3),
                            loc=(i // 3, i % 3),
                            colspan=1,
                            rowspan=1,
                            projection='3d' if dim == 3 else None
                        )
                    )

                plot_points(result, bounds, axes)
            else:
                print('Построить точки можно только для 2d и 3d областей')

        if surface:
            if dim == 1 or dim == 2:
                axes.append(
                    plt.subplot2grid(
                        shape=(nrows, 3),
                        loc=(nrows - 1, 0),
                        colspan=3,
                        rowspan=1,
                        projection='3d' if dim == 2 else None
                    )
                )

                plot_surface(name, bounds, axes[-1], dim)
            else:
                print('\nПредупреждение: Построить функцию можно только для 2d и 3d!\n')

        plt.show()
