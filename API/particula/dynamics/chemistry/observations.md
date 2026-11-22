# particula.dynamics.chemistry.observations

*module*

Time-resolved observation likelihoods for kinetic mechanism discovery.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/observations.py#L1-L1)

## particula.dynamics.chemistry.observations.ObservationSeries

*class* · *dataclass*

```python
class ObservationSeries
```

Finite observations and whitening factors at irregular sample times.

Covariances have squared or cross-product measurement units. Their
diagonals describe variances. Off-diagonal entries represent correlated
channels at the same time. A missing observation is represented by NaN.
Each observed block must be positive definite: omit one channel for an
exact derived total. Distinct time points are treated as independent.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/observations.py#L10-L73)

### particula.dynamics.chemistry.observations.ObservationSeries.factors

*attribute* · *instance attribute*

```python
factors: tuple[NDArray[np.float64], ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/observations.py#L24-L24)

### particula.dynamics.chemistry.observations.ObservationSeries.indices

*attribute* · *instance attribute*

```python
indices: tuple[NDArray[np.intp], ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/observations.py#L23-L23)

### particula.dynamics.chemistry.observations.ObservationSeries.times

*attribute* · *instance attribute*

```python
times: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/observations.py#L21-L21)

### particula.dynamics.chemistry.observations.ObservationSeries.values

*attribute* · *instance attribute*

```python
values: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/observations.py#L22-L22)

### particula.dynamics.chemistry.observations.ObservationSeries.__init__

*method*

```python
def __init__(times: NDArray[np.float64], values: NDArray[np.float64], indices: tuple[NDArray[np.intp], ...], factors: tuple[NDArray[np.float64], ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/observations.py#L0-L0)

### particula.dynamics.chemistry.observations.ObservationSeries.create

*method* · *classmethod*

```python
def create(times: ArrayLike, values: ArrayLike, covariance: ArrayLike)
```

Validate observations and shared or per-time covariance.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/observations.py#L26-L54)

### particula.dynamics.chemistry.observations.ObservationSeries.loss

*method*

```python
def loss(predictions: ArrayLike) -> tuple[float, NDArray[np.float64]]
```

Evaluate the Gaussian loss and its prediction gradient.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/observations.py#L56-L73)

## particula.dynamics.chemistry.observations._factor

*function*

```python
def _factor(covariance)
```

Check the observed covariance before using its lower triangle.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/observations.py#L76-L87)
