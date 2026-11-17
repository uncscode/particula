# particula.execution.forcing

*module*

Immutable environmental forcing with explicit one-sided event semantics.

A table is right-continuous at knots. An integration interval freezes the
selected segment, so endpoint stages before a jump use the left-hand physics.
An accepted snapshot at the knot uses the right-hand environment. Event position
is derived from absolute time and needs no mutable restart cursor.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L1-L1)

## particula.execution.forcing.EnvironmentForcing

*class* · *dataclass*

```python
class EnvironmentForcing
```

Declare temperature [K] and pressure [Pa] at absolute times [s].

times starts at zero and increases strictly. Each value input may be a
scalar, a vector with one value per time, or a time-by-box matrix. A width
of one broadcasts across boxes. mode='step' holds each row to the next knot;
mode='linear' interpolates. The last row is held indefinitely. Inputs are
copied into immutable tuples; no user callbacks or file-named code is used.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L16-L109)

### particula.execution.forcing.EnvironmentForcing.mode

*attribute* · *class attribute* · *instance attribute*

```python
mode: str = 'linear'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L30-L30)

### particula.execution.forcing.EnvironmentForcing.pressure

*attribute* · *instance attribute*

```python
pressure: float | tuple | list | np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L29-L29)

### particula.execution.forcing.EnvironmentForcing.temperature

*attribute* · *instance attribute*

```python
temperature: float | tuple | list | np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L28-L28)

### particula.execution.forcing.EnvironmentForcing.times

*attribute* · *instance attribute*

```python
times: tuple | list | np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L27-L27)

### particula.execution.forcing.EnvironmentForcing.__call__

*method*

```python
def __call__(time)
```

Return right-continuous environmental values at absolute time.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L96-L98)

### particula.execution.forcing.EnvironmentForcing.__init__

*method*

```python
def __init__(times: tuple | list | np.ndarray, temperature: float | tuple | list | np.ndarray, pressure: float | tuple | list | np.ndarray, mode: str = 'linear') -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L0-L0)

### particula.execution.forcing.EnvironmentForcing.__post_init__

*method*

```python
def __post_init__()
```

Validate finite positive environmental rows and freeze their data.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L32-L67)

### particula.execution.forcing.EnvironmentForcing._index

*method*

```python
def _index(time)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L69-L72)

### particula.execution.forcing.EnvironmentForcing._values

*method*

```python
def _values(index, time)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L74-L94)

### particula.execution.forcing.EnvironmentForcing.next_event

*method*

```python
def next_event(time)
```

Return the next knot strictly after time, or infinity.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L100-L103)

### particula.execution.forcing.EnvironmentForcing.segment

*method*

```python
def segment(time)
```

Freeze one segment through its endpoint for all trial stages.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L105-L109)

## particula.execution.forcing._EnvironmentSegment

*class* · *dataclass*

```python
class _EnvironmentSegment
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L112-L119)

### particula.execution.forcing._EnvironmentSegment.end

*attribute* · *instance attribute*

```python
end: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L116-L116)

### particula.execution.forcing._EnvironmentSegment.index

*attribute* · *instance attribute*

```python
index: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L115-L115)

### particula.execution.forcing._EnvironmentSegment.table

*attribute* · *instance attribute*

```python
table: EnvironmentForcing
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L114-L114)

### particula.execution.forcing._EnvironmentSegment.__call__

*method*

```python
def __call__(time)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L118-L119)

### particula.execution.forcing._EnvironmentSegment.__init__

*method*

```python
def __init__(table: EnvironmentForcing, index: int, end: float) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L0-L0)

## particula.execution.forcing.checkpoint_forcing

*function*

```python
def checkpoint_forcing(forcing)
```

Admit only immutable tables as exact restart data.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L136-L140)

## particula.execution.forcing.forcing_interval

*function*

```python
def forcing_interval(owner, bound)
```

Bind interval stages separately from accepted observation forcing.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/forcing.py#L122-L133)
