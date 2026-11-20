# particula.dynamics.chemistry.resolved_device

*module*

Resident RK23 for resolved gas and particle material balance.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L1-L1)

## particula.dynamics.chemistry.resolved_device.NEGATIVE_THIRD

*attribute* · *module attribute*

```python
NEGATIVE_THIRD = wp.constant(wp.float64(-1 / 3))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L14-L14)

## particula.dynamics.chemistry.resolved_device.PI

*attribute* · *module attribute*

```python
PI = wp.constant(wp.float64(3.141592653589793))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L12-L12)

## particula.dynamics.chemistry.resolved_device.R

*attribute* · *module attribute*

```python
R = wp.constant(wp.float64(GAS_CONSTANT))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L11-L11)

## particula.dynamics.chemistry.resolved_device.SAFETY

*attribute* · *module attribute*

```python
SAFETY = wp.constant(wp.float64(0.9))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L15-L15)

## particula.dynamics.chemistry.resolved_device.THIRD

*attribute* · *module attribute*

```python
THIRD = wp.constant(wp.float64(1 / 3))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L13-L13)

## particula.dynamics.chemistry.resolved_device.Network

*class*

```python
class Network
```

Immutable reaction and solver configuration shared by boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L18-L37)

### particula.dynamics.chemistry.resolved_device.Network.atol

*attribute* · *instance attribute*

```python
atol: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L32-L32)

### particula.dynamics.chemistry.resolved_device.Network.coefficients_net

*attribute* · *instance attribute*

```python
coefficients_net: wp.array(dtype=int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L27-L27)

### particula.dynamics.chemistry.resolved_device.Network.columns

*attribute* · *instance attribute*

```python
columns: wp.array(dtype=int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L26-L26)

### particula.dynamics.chemistry.resolved_device.Network.density

*attribute* · *instance attribute*

```python
density: wp.array(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L24-L24)

### particula.dynamics.chemistry.resolved_device.Network.first

*attribute* · *instance attribute*

```python
first: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L34-L34)

### particula.dynamics.chemistry.resolved_device.Network.gm

*attribute* · *instance attribute*

```python
gm: wp.array(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L22-L22)

### particula.dynamics.chemistry.resolved_device.Network.kinds

*attribute* · *instance attribute*

```python
kinds: wp.array(dtype=int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L30-L30)

### particula.dynamics.chemistry.resolved_device.Network.limit

*attribute* · *instance attribute*

```python
limit: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L37-L37)

### particula.dynamics.chemistry.resolved_device.Network.maximum

*attribute* · *instance attribute*

```python
maximum: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L36-L36)

### particula.dynamics.chemistry.resolved_device.Network.minimum

*attribute* · *instance attribute*

```python
minimum: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L35-L35)

### particula.dynamics.chemistry.resolved_device.Network.offsets

*attribute* · *instance attribute*

```python
offsets: wp.array(dtype=int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L25-L25)

### particula.dynamics.chemistry.resolved_device.Network.orders

*attribute* · *instance attribute*

```python
orders: wp.array2d(dtype=int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L28-L28)

### particula.dynamics.chemistry.resolved_device.Network.params

*attribute* · *instance attribute*

```python
params: wp.array2d(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L29-L29)

### particula.dynamics.chemistry.resolved_device.Network.pm

*attribute* · *instance attribute*

```python
pm: wp.array(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L23-L23)

### particula.dynamics.chemistry.resolved_device.Network.rtol

*attribute* · *instance attribute*

```python
rtol: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L33-L33)

### particula.dynamics.chemistry.resolved_device.Network.sources

*attribute* · *instance attribute*

```python
sources: wp.array(dtype=int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L31-L31)

## particula.dynamics.chemistry.resolved_device.Stages

*class*

```python
class Stages
```

Work vectors hold accepted state, trial state and four derivatives.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L40-L48)

### particula.dynamics.chemistry.resolved_device.Stages.counts

*attribute* · *instance attribute*

```python
counts: wp.array2d(dtype=wp.int64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L48-L48)

### particula.dynamics.chemistry.resolved_device.Stages.rates

*attribute* · *instance attribute*

```python
rates: wp.array2d(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L47-L47)

### particula.dynamics.chemistry.resolved_device.Stages.reference

*attribute* · *instance attribute*

```python
reference: wp.array2d(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L45-L45)

### particula.dynamics.chemistry.resolved_device.Stages.v

*attribute* · *instance attribute*

```python
v: wp.array3d(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L44-L44)

### particula.dynamics.chemistry.resolved_device.Stages.weights

*attribute* · *instance attribute*

```python
weights: wp.array2d(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L46-L46)

## particula.dynamics.chemistry.resolved_device._evaluate

*function*

```python
def _evaluate(m: Network, w: Stages, b: int, target: int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L51-L106)

## particula.dynamics.chemistry.resolved_device._integrate

*function*

```python
def _integrate(m: Network, w: Stages, b: int, dt: wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L133-L180)

## particula.dynamics.chemistry.resolved_device._trial

*function*

```python
def _trial(m: Network, w: Stages, b: int, h: wp.float64)
```

Bogacki–Shampine stages with a common step for both reservoirs.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L109-L130)

## particula.dynamics.chemistry.resolved_device.publish

*function*

```python
def publish(m: Network, w: Stages, gas: wp.array2d(dtype=wp.float64), particles: wp.array3d(dtype=wp.float64), failure: wp.array(dtype=int))
```

Commit each changed lane only after all boxes succeeded.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L236-L258)

## particula.dynamics.chemistry.resolved_device.stage

*function*

```python
def stage(m: Network, w: Stages, gas: wp.array2d(dtype=wp.float64), particles: wp.array3d(dtype=wp.float64), counts: wp.array2d(dtype=wp.float64), volume: wp.array(dtype=wp.float64), temp: wp.array(dtype=wp.float64), dt: wp.float64, failure: wp.array(dtype=int))
```

Advance one thread per box into unpublished scratch.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/resolved_device.py#L183-L233)
