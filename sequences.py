from chaospy import J, Uniform
from numpy import newaxis

sequence_names = (
    'random',
    'additive_recursion',
    'halton', 'hammersley',
    'korobov',
    'sobol',
    'latin_hypercube',
)


def check_sequence(sequence_name):
    assert sequence_name in sequence_names, f'Последовательность "{sequence_name}" не поддерживается'


def seq(**kwargs):
    sequence_name = kwargs.get('sequence_name')
    check_sequence(sequence_name)

    dim = kwargs.get('dim') or 2
    n = kwargs.get('n') or 5
    bounds = kwargs.get('bounds') or [[0, 1] for _ in range(dim)]

    uniform_cube = J(*[Uniform(*bound) for bound in bounds])

    sample = uniform_cube.sample(n, rule=sequence_name)

    if sample.ndim == 1:
        return sample[newaxis].T

    return sample.T
