# particula.dynamics.chemistry.inference

*module*

Fit mechanism candidates to experiments with shared or local parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L1-L1)

## particula.dynamics.chemistry.inference.Loss

*attribute* · *module attribute*

```python
Loss = Callable[[Vector, bool], tuple[float, Vector | None]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L13-L13)

## particula.dynamics.chemistry.inference.Vector

*attribute* · *module attribute*

```python
Vector = NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L12-L12)

## particula.dynamics.chemistry.inference.CandidateFit

*class* · *dataclass*

```python
class CandidateFit
```

Calibration and held-out experiments for one candidate.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L165-L172)

### particula.dynamics.chemistry.inference.CandidateFit.bounds

*attribute* · *instance attribute*

```python
bounds: Sequence[tuple[float, float]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L170-L170)

### particula.dynamics.chemistry.inference.CandidateFit.experiments

*attribute* · *instance attribute*

```python
experiments: Sequence[ExperimentLoss]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L169-L169)

### particula.dynamics.chemistry.inference.CandidateFit.held_out

*attribute* · *class attribute* · *instance attribute*

```python
held_out: Sequence[ExperimentLoss] = ()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L172-L172)

### particula.dynamics.chemistry.inference.CandidateFit.starts

*attribute* · *instance attribute*

```python
starts: Sequence[Vector]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L171-L171)

### particula.dynamics.chemistry.inference.CandidateFit.__init__

*method*

```python
def __init__(experiments: Sequence[ExperimentLoss], bounds: Sequence[tuple[float, float]], starts: Sequence[Vector], held_out: Sequence[ExperimentLoss] = ()) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L0-L0)

## particula.dynamics.chemistry.inference.ExperimentLoss

*class* · *dataclass*

```python
class ExperimentLoss
```

Experiment objective and its mapping into the global parameter vector.

Reusing an index across experiments shares the corresponding physical
parameter. Using a distinct index makes it experiment-specific. The
objective receives local physical values and a gradient request.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L16-L26)

### particula.dynamics.chemistry.inference.ExperimentLoss.evaluate

*attribute* · *instance attribute*

```python
evaluate: Loss
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L26-L26)

### particula.dynamics.chemistry.inference.ExperimentLoss.parameter_indices

*attribute* · *instance attribute*

```python
parameter_indices: tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L25-L25)

### particula.dynamics.chemistry.inference.ExperimentLoss.__init__

*method*

```python
def __init__(parameter_indices: tuple[int, ...], evaluate: Loss) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L0-L0)

## particula.dynamics.chemistry.inference.FitResult

*class* · *dataclass*

```python
class FitResult
```

Physical parameters, loss and diagnostics for each start.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L29-L36)

### particula.dynamics.chemistry.inference.FitResult.converged

*attribute* · *instance attribute*

```python
converged: tuple[bool, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L35-L35)

### particula.dynamics.chemistry.inference.FitResult.evaluations

*attribute* · *instance attribute*

```python
evaluations: tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L36-L36)

### particula.dynamics.chemistry.inference.FitResult.loss

*attribute* · *instance attribute*

```python
loss: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L34-L34)

### particula.dynamics.chemistry.inference.FitResult.parameters

*attribute* · *instance attribute*

```python
parameters: Vector
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L33-L33)

### particula.dynamics.chemistry.inference.FitResult.__init__

*method*

```python
def __init__(parameters: Vector, loss: float, converged: tuple[bool, ...], evaluations: tuple[int, ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L0-L0)

## particula.dynamics.chemistry.inference.RankedNetwork

*class* · *dataclass*

```python
class RankedNetwork
```

Mechanism and fit, with score gap measured from the best candidate.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L175-L183)

### particula.dynamics.chemistry.inference.RankedNetwork.competitive

*attribute* · *instance attribute*

```python
competitive: bool
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L183-L183)

### particula.dynamics.chemistry.inference.RankedNetwork.delta

*attribute* · *instance attribute*

```python
delta: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L182-L182)

### particula.dynamics.chemistry.inference.RankedNetwork.fit

*attribute* · *instance attribute*

```python
fit: FitResult
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L180-L180)

### particula.dynamics.chemistry.inference.RankedNetwork.network

*attribute* · *instance attribute*

```python
network: Network
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L179-L179)

### particula.dynamics.chemistry.inference.RankedNetwork.score

*attribute* · *instance attribute*

```python
score: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L181-L181)

### particula.dynamics.chemistry.inference.RankedNetwork.__init__

*method*

```python
def __init__(network: Network, fit: FitResult, score: float, delta: float, competitive: bool) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L0-L0)

## particula.dynamics.chemistry.inference._minimize

*function*

```python
def _minimize(objective, initial, gradient, max_iterations, max_evaluations)
```

Enforce the same objective-call ceiling for AD and finite differences.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L64-L91)

## particula.dynamics.chemistry.inference._parameter_domain

*function*

```python
def _parameter_domain(bounds, starts)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L39-L61)

## particula.dynamics.chemistry.inference.fit_experiments

*function*

```python
def fit_experiments(experiments: Sequence[ExperimentLoss], bounds: Sequence[tuple[float, float]], starts: Sequence[Vector], *, gradient: bool = True, max_iterations: int = 100, max_evaluations: int = 1000) -> FitResult
```

Fit finite box-bounded parameters in normalized linear coordinates.

Zero-valued rates are feasible when the supplied lower bound is zero.
Set the bounds equal to fix a parameter for a rate profile or channel
deletion control. Finite-difference fits use the same bounds and starts.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L94-L162)

## particula.dynamics.chemistry.inference.fit_networks

*function*

```python
def fit_networks(search: SearchResult, prepare: Callable[[Network], CandidateFit], *, criterion: str, tolerance: float, gradient=True, max_iterations=100, max_evaluations=1000) -> tuple[RankedNetwork, ...]
```

Rank every enumerated candidate using matched fitting budgets.

For 'aic', each fit objective must be a negative log likelihood for the
same observations and noise model. Count all free fitted parameters.
For 'held_out', score untouched experiments at training estimates.
The tolerance is in the selected score's units. All candidates within
it remain competitive, including simpler parents. Search completeness
and its stopping reason remain available on the supplied SearchResult.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/inference.py#L186-L253)
