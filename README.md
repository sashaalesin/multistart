# multistart

```bash
pipenv install
python multistart.py
```

## Algorithm:

1. Modeling the distribution on X, we obtain N points (x1,..., XN);
2. Using the points (x1,..., XN) as initial, we carry out one or several iterations of the local minimum search algorithm, we obtain the points (z1,..., ZN);
3. Apply the clustering method to points (z1,..., ZN). Let m be the number of resulting clusters. If m = 1, then go to step 5, otherwise - to step 4;
4. Select representatives (x1,..., Xm) from each cluster (select points with the smallest value of the objective function). We set N = m and go to step 2;
5. We assume that we are in the neighbourhood of the global minimum point. We select a representative from a single cluster. We will use it as a starting point for the local minimization algorithm.

### Point generation

- 'random'
- 'additive_recursion'
- 'halton' 'hammersley'
- 'korobov'
- 'sobol'
- 'latin_hypercube'

Samples are provided by [chaospy](https://chaospy.readthedocs.io/en/master/user_guide/fundamentals/quasi_random_samples.html).

Adding new generation algorithms to `sequences.py` is welcome.

### Test functions for optimization

- Sphere
- Rastrigin
- Rosenbrock
- Himmelblau's
- Easom
- Ackley

Adding new test functions to `test_functions.py` is welcome.

### Parameters

```python
# Dimension of points (dimension of the function is 1 higher)
dim = 1
# Function (set in test_functions.py)
func = Rastrigin(dim)
# Sequence (set in sequences.py)
sequence_name = 'halton'

result = multistart(
    func=func,
    sequence_name=sequence_name,
    dim=dim,
    # Number of points
    n=32,
    # Search area (set in test_functions.py or [[0, 1], ...])
    bounds=None,
    # Initial number of clusters
    n_clusters=10,
    # Points and clusters visualization (only for dim=1,2,3)
    plot_points=True,
    # Function visualization (only for dim=1,2)
    plot_surface=True,
    # Console output
    print=True,
)
```

### Results

- Console output (`print=True`)
- Visualization of points and clusters at each iteration. Available only for 1-3 dimensions (`plot_points=True`)
- Function visualization. Available only for 1-2 dimensions (`plot_surface=True`)
