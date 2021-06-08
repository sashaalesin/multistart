import numpy as np
import sobol_seq


def set_seq_to_bounds(seq, bounds):
    # Накладывает последовательность в [0, 1] x [0, 1] на bounds
    return np.column_stack([
        (bounds[i][1] - bounds[i][0]) * seq[:, i] + bounds[i][0]
        for i in range(len(bounds))
    ])


def primes_from_2_to(n):
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def van_der_corput(n_sample, base=2):
    sequence = []
    for i in range(n_sample):
        n_th_number, denom = 0., 1.
        while i > 0:
            i, remainder = divmod(i, base)
            denom *= base
            n_th_number += remainder / denom
        sequence.append(n_th_number)

    return sequence


def halton(**kwargs):
    dim = kwargs.get('dim') or 2
    n = kwargs.get('n') or 5
    bounds = kwargs.get('bounds') or [[0, 1] for _ in range(dim)]

    big_number = 10
    while 'Not enought primes':
        base = primes_from_2_to(big_number)[:dim]
        if len(base) == dim:
            break
        big_number += 1000

    sample = [van_der_corput(n + 1, dim) for dim in base]
    sample = np.stack(sample, axis=-1)[1:]

    return set_seq_to_bounds(sample, bounds)


def sobol(**kwargs):
    dim = kwargs.get('dim') or 2
    n = kwargs.get('n') or 5
    bounds = kwargs.get('bounds') or [[0, 1] for _ in range(dim)]
    return set_seq_to_bounds(sobol_seq.i4_sobol_generate(dim, n), bounds)


def sobol_quasi_random(dim, n_sample):
    return sobol_seq.i4_sobol_generate_std_normal(dim, n_sample)


def random(**kwargs):
    dim = kwargs.get('dim') or 2
    n = kwargs.get('n') or 5
    bounds = kwargs.get('bounds') or [[0, 1] for _ in range(dim)]
    return set_seq_to_bounds(np.random.random_sample((n, dim)), bounds)


s = {
    'random': random,
    'sobol': sobol,
    'halton': halton,
}
