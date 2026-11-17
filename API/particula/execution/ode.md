# particula.execution.ode

*module*

Concrete deterministic ODE stepping, separate from runnable finite-step maps.

Rates have units of state per second. Callbacks receive detached read-only
arrays and must be pure: no RNG, observation, or external mutation. The caller
owns state scaling, physical admissibility and linear conservation invariants.
Bogacki-Shampine RK3(2) is for nonstiff rates; ROS-2 uses the KPP coefficients:
https://kpp.readthedocs.io/en/stable/num_methods/rosenbrock-methods.html
These names are concrete implementation APIs, not package-root exports.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L1-L1)

## particula.execution.ode.Array

*attribute* · *module attribute*

```python
Array = np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L17-L17)

## particula.execution.ode.Rate

*attribute* · *module attribute*

```python
Rate = Callable[[float, Array], Array]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L18-L18)

## particula.execution.ode.AdaptiveSolver

*class*

```python
class AdaptiveSolver
```

Own accepted ODE state and advance one error-controlled step at a time.

Errors use max(abs(error)/(atol + rtol*max(abs(old),abs(new)))). Scalars
or per-component tolerances use the state units, with strictly positive
atol. Domain rejection reduces the step without clipping inventories.
A failed call preserves the last accepted state/time, but retains work
counters. Earlier accepted steps are never rolled back. Output/restart
orchestration observes this owner only between successful step() calls.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L61-L266)

### particula.execution.ode.AdaptiveSolver._state

*attribute* · *instance attribute*

```python
_state = value
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L103-L103)

### particula.execution.ode.AdaptiveSolver._time

*attribute* · *instance attribute*

```python
_time = self._scalar(time, 'time', positive=False)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L104-L104)

### particula.execution.ode.AdaptiveSolver.atol

*attribute* · *instance attribute*

```python
atol = self._tolerance(atol, value.shape, 'atol', True)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L118-L118)

### particula.execution.ode.AdaptiveSolver.last_error

*attribute* · *instance attribute*

```python
last_error = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L121-L121)

### particula.execution.ode.AdaptiveSolver.max_step

*attribute* · *instance attribute*

```python
max_step = self._scalar(max_step, 'max_step')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L106-L106)

### particula.execution.ode.AdaptiveSolver.max_trials

*attribute* · *instance attribute*

```python
max_trials = max_trials
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L116-L116)

### particula.execution.ode.AdaptiveSolver.method

*attribute* · *instance attribute*

```python
method = method
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L101-L101)

### particula.execution.ode.AdaptiveSolver.min_step

*attribute* · *instance attribute*

```python
min_step = self._scalar(min_step, 'min_step')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L105-L105)

### particula.execution.ode.AdaptiveSolver.next_step

*attribute* · *instance attribute*

```python
next_step = first_step
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L119-L119)

### particula.execution.ode.AdaptiveSolver.problem

*attribute* · *instance attribute*

```python
problem = problem
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L100-L100)

### particula.execution.ode.AdaptiveSolver.rtol

*attribute* · *instance attribute*

```python
rtol = self._tolerance(rtol, value.shape, 'rtol', False)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L117-L117)

### particula.execution.ode.AdaptiveSolver.statistics

*attribute* · *instance attribute*

```python
statistics = StepStatistics()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L120-L120)

### particula.execution.ode.AdaptiveSolver.state

*property*

```python
state
```

Return detached accepted state in the caller's declared units.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L144-L146)

### particula.execution.ode.AdaptiveSolver.time

*property*

```python
time
```

Return accepted absolute time in seconds.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L149-L151)

### particula.execution.ode.AdaptiveSolver.__init__

*method*

```python
def __init__(problem, state, *, time=0.0, method='rk23', rtol=1e-05, atol=1e-12, first_step=0.001, min_step=1e-14, max_step=1.0, max_trials=100)
```

Validate configuration and detach initial state before any rates.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L72-L121)

### particula.execution.ode.AdaptiveSolver._check

*method*

```python
def _check(state)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L153-L157)

### particula.execution.ode.AdaptiveSolver._rate

*method*

```python
def _rate(time, state)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L159-L169)

### particula.execution.ode.AdaptiveSolver._rk23

*method*

```python
def _rk23(t, y, h)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L171-L178)

### particula.execution.ode.AdaptiveSolver._ros2

*method*

```python
def _ros2(t, y, h)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L180-L207)

### particula.execution.ode.AdaptiveSolver._scalar

*method* · *staticmethod*

```python
def _scalar(value, name, positive=True)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L123-L130)

### particula.execution.ode.AdaptiveSolver._tolerance

*method* · *staticmethod*

```python
def _tolerance(value, shape, name, positive)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L132-L141)

### particula.execution.ode.AdaptiveSolver.step

*method*

```python
def step(bound)
```

Accept one step no later than bound [s]; return its accepted time.

At an already reached bound this is a no-op. Only explicit domain
failures are retried; configuration errors and callback bugs propagate.
Repeated rejection or unrepresentable time advance raises RuntimeError.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L209-L266)

## particula.execution.ode.ODEProblem

*class* · *dataclass*

```python
class ODEProblem
```

Declare pure continuous derivatives and optional stiff-method inputs.

jacobian[i,j] is df_i/dy_j. ROS-2 requires both jacobian and
time_derivative (partial f/partial t at fixed state); autonomous problems
explicitly supply zeros for the latter. admissible checks all internal
stages and the accepted candidate, without changing either.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L25-L38)

### particula.execution.ode.ODEProblem.admissible

*attribute* · *instance attribute*

```python
admissible: Callable[[Array], bool]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L36-L36)

### particula.execution.ode.ODEProblem.jacobian

*attribute* · *class attribute* · *instance attribute*

```python
jacobian: Rate | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L37-L37)

### particula.execution.ode.ODEProblem.rate

*attribute* · *instance attribute*

```python
rate: Rate
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L35-L35)

### particula.execution.ode.ODEProblem.time_derivative

*attribute* · *class attribute* · *instance attribute*

```python
time_derivative: Rate | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L38-L38)

### particula.execution.ode.ODEProblem.__init__

*method*

```python
def __init__(rate: Rate, admissible: Callable[[Array], bool], jacobian: Rate | None = None, time_derivative: Rate | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L0-L0)

## particula.execution.ode.StepStatistics

*class* · *dataclass*

```python
class StepStatistics
```

Count trial work separately from accepted steps.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L41-L52)

### particula.execution.ode.StepStatistics.accepted

*attribute* · *class attribute* · *instance attribute*

```python
accepted: int = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L45-L45)

### particula.execution.ode.StepStatistics.domain_rejected

*attribute* · *class attribute* · *instance attribute*

```python
domain_rejected: int = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L47-L47)

### particula.execution.ode.StepStatistics.factorizations

*attribute* · *class attribute* · *instance attribute*

```python
factorizations: int = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L51-L51)

### particula.execution.ode.StepStatistics.jacobian_calls

*attribute* · *class attribute* · *instance attribute*

```python
jacobian_calls: int = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L49-L49)

### particula.execution.ode.StepStatistics.linear_solves

*attribute* · *class attribute* · *instance attribute*

```python
linear_solves: int = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L52-L52)

### particula.execution.ode.StepStatistics.rate_calls

*attribute* · *class attribute* · *instance attribute*

```python
rate_calls: int = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L48-L48)

### particula.execution.ode.StepStatistics.rejected

*attribute* · *class attribute* · *instance attribute*

```python
rejected: int = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L46-L46)

### particula.execution.ode.StepStatistics.time_derivative_calls

*attribute* · *class attribute* · *instance attribute*

```python
time_derivative_calls: int = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L50-L50)

### particula.execution.ode.StepStatistics.__init__

*method*

```python
def __init__(accepted: int = 0, rejected: int = 0, domain_rejected: int = 0, rate_calls: int = 0, jacobian_calls: int = 0, time_derivative_calls: int = 0, factorizations: int = 0, linear_solves: int = 0) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L0-L0)

## particula.execution.ode.TrialDomainError

*class*

```python
class TrialDomainError(ValueError)
```

Bases: `ValueError`

A proposed intermediate state is outside the model's physical domain.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L21-L22)

## particula.execution.ode._readonly

*function*

```python
def _readonly(value)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/ode.py#L55-L58)
