from numpy import round


def r(point):
    return round(point, 5)


def print_result(result):
    print('\nНачальные точки:')
    for point in result[0]['points']:
        print(f'\t{r(point)}')
    number_of_iteration = result['numberOfIteration']
    for i in range(number_of_iteration):
        v = result[i + 1]
        print(f'\n\nИтерация {i + 1}')
        print('\n\tЛокальные минимумы:\n')
        for point in v['localMinimums']:
            print(f'\t\t{r(point)}')
        print('\n\tКластеры:')
        for idx, cluster in enumerate(
            v['clusteredLocalMinimums']['clusters']
        ):
            print(f'\n\tКластер {idx + 1}:\n')
            for point in cluster:
                print(f'\t\t{r(point)}')
        print('\n\tВыбранные минимальные точки:\t')
        for point in v['clusteredLocalMinimums']['minPoints']:
            print(f'\t\t{r(point)}')

    print(f'\nЧисло итераций: {number_of_iteration}')
    print(f"\nОтвет: {result['answer']}\n")
