# particula.dynamics.chemistry.device

*module*

Double-precision resident mass-action RK23 and ROS3 box integrators.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L1-L1)

## particula.dynamics.chemistry.device.DELTA

*attribute* · *module attribute*

```python
DELTA = wp.constant(8)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L64-L64)

## particula.dynamics.chemistry.device.ERROR

*attribute* · *module attribute*

```python
ERROR = wp.constant(6)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L62-L62)

## particula.dynamics.chemistry.device.K1

*attribute* · *module attribute*

```python
K1 = wp.constant(2)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L58-L58)

## particula.dynamics.chemistry.device.K2

*attribute* · *module attribute*

```python
K2 = wp.constant(3)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L59-L59)

## particula.dynamics.chemistry.device.K3

*attribute* · *module attribute*

```python
K3 = wp.constant(4)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L60-L60)

## particula.dynamics.chemistry.device.K4

*attribute* · *module attribute*

```python
K4 = wp.constant(5)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L61-L61)

## particula.dynamics.chemistry.device.MOLAR_GAS_CONSTANT

*attribute* · *module attribute*

```python
MOLAR_GAS_CONSTANT = wp.constant(wp.float64(GAS_CONSTANT))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L65-L65)

## particula.dynamics.chemistry.device.POWER

*attribute* · *module attribute*

```python
POWER = wp.constant(wp.float64(-1.0 / 3.0))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L67-L67)

## particula.dynamics.chemistry.device.RHS

*attribute* · *module attribute*

```python
RHS = wp.constant(7)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L63-L63)

## particula.dynamics.chemistry.device.SAFETY

*attribute* · *module attribute*

```python
SAFETY = wp.constant(wp.float64(0.9))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L66-L66)

## particula.dynamics.chemistry.device.STAGE

*attribute* · *module attribute*

```python
STAGE = wp.constant(1)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L57-L57)

## particula.dynamics.chemistry.device.Y

*attribute* · *module attribute*

```python
Y = wp.constant(0)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L56-L56)

## particula.dynamics.chemistry.device.DeviceMechanism

*class*

```python
class DeviceMechanism
```

Immutable resident configuration for all boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L22-L40)

### particula.dynamics.chemistry.device.DeviceMechanism.active

*attribute* · *instance attribute*

```python
active: wp.array(dtype=int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L28-L28)

### particula.dynamics.chemistry.device.DeviceMechanism.atol

*attribute* · *instance attribute*

```python
atol: wp.array(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L34-L34)

### particula.dynamics.chemistry.device.DeviceMechanism.coefficients

*attribute* · *instance attribute*

```python
coefficients: wp.array2d(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L33-L33)

### particula.dynamics.chemistry.device.DeviceMechanism.coefficients_net

*attribute* · *instance attribute*

```python
coefficients_net: wp.array(dtype=int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L31-L31)

### particula.dynamics.chemistry.device.DeviceMechanism.columns

*attribute* · *instance attribute*

```python
columns: wp.array(dtype=int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L30-L30)

### particula.dynamics.chemistry.device.DeviceMechanism.first_step

*attribute* · *instance attribute*

```python
first_step: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L36-L36)

### particula.dynamics.chemistry.device.DeviceMechanism.lanes

*attribute* · *instance attribute*

```python
lanes: wp.array(dtype=int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L26-L26)

### particula.dynamics.chemistry.device.DeviceMechanism.masses

*attribute* · *instance attribute*

```python
masses: wp.array(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L27-L27)

### particula.dynamics.chemistry.device.DeviceMechanism.max_step

*attribute* · *instance attribute*

```python
max_step: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L38-L38)

### particula.dynamics.chemistry.device.DeviceMechanism.min_step

*attribute* · *instance attribute*

```python
min_step: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L37-L37)

### particula.dynamics.chemistry.device.DeviceMechanism.offsets

*attribute* · *instance attribute*

```python
offsets: wp.array(dtype=int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L29-L29)

### particula.dynamics.chemistry.device.DeviceMechanism.orders

*attribute* · *instance attribute*

```python
orders: wp.array2d(dtype=int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L32-L32)

### particula.dynamics.chemistry.device.DeviceMechanism.rtol

*attribute* · *instance attribute*

```python
rtol: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L35-L35)

### particula.dynamics.chemistry.device.DeviceMechanism.stiff

*attribute* · *instance attribute*

```python
stiff: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L40-L40)

### particula.dynamics.chemistry.device.DeviceMechanism.trials

*attribute* · *instance attribute*

```python
trials: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L39-L39)

## particula.dynamics.chemistry.device.Work

*class*

```python
class Work
```

Fixed scratch indexed by box, plus interval work counters.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L43-L52)

### particula.dynamics.chemistry.device.Work.counts

*attribute* · *instance attribute*

```python
counts: wp.array2d(dtype=wp.int64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L52-L52)

### particula.dynamics.chemistry.device.Work.lu

*attribute* · *instance attribute*

```python
lu: wp.array3d(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L49-L49)

### particula.dynamics.chemistry.device.Work.matrix

*attribute* · *instance attribute*

```python
matrix: wp.array3d(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L48-L48)

### particula.dynamics.chemistry.device.Work.pivots

*attribute* · *instance attribute*

```python
pivots: wp.array2d(dtype=int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L50-L50)

### particula.dynamics.chemistry.device.Work.rates

*attribute* · *instance attribute*

```python
rates: wp.array2d(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L51-L51)

### particula.dynamics.chemistry.device.Work.vectors

*attribute* · *instance attribute*

```python
vectors: wp.array3d(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L47-L47)

## particula.dynamics.chemistry.device._factor

*function*

```python
def _factor(m: DeviceMechanism, w: Work, b: int, h: wp.float64)
```

Form I/(gamma*h) - J and factor it with partial pivoting.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L149-L184)

## particula.dynamics.chemistry.device._product

*function*

```python
def _product(m: DeviceMechanism, w: Work, box: int, slot: int, reaction: int, derivative: int)
```

Evaluate a molar rate or a partial without division by concentration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L70-L90)

## particula.dynamics.chemistry.device._rate

*function*

```python
def _rate(m: DeviceMechanism, w: Work, box: int, slot: int, target: int)
```

Sum all molar reaction extents into one species tendency.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L93-L110)

## particula.dynamics.chemistry.device._rk23

*function*

```python
def _rk23(m: DeviceMechanism, w: Work, b: int, h: wp.float64)
```

Bogacki–Shampine embedded 3(2) trial, matching the CPU tableau.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L113-L146)

## particula.dynamics.chemistry.device._ros3

*function*

```python
def _ros3(m: DeviceMechanism, w: Work, b: int, h: wp.float64)
```

Autonomous ROS3 step at fixed temperature, sharing one factorization.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L217-L252)

## particula.dynamics.chemistry.device._solve

*function*

```python
def _solve(m: DeviceMechanism, w: Work, b: int, target: int)
```

Solve a stage and perform the CPU method's two residual corrections.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L187-L214)

## particula.dynamics.chemistry.device.integrate_boxes

*function*

```python
def integrate_boxes(m: DeviceMechanism, w: Work, gas: wp.array2d(dtype=wp.float64), temperature: wp.array(dtype=wp.float64), dt: wp.float64, failure: wp.array(dtype=int))
```

Stage each box independently and signal a failed interval on device.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L255-L329)

## particula.dynamics.chemistry.device.publish_boxes

*function*

```python
def publish_boxes(m: DeviceMechanism, w: Work, gas: wp.array2d(dtype=wp.float64), failure: wp.array(dtype=int))
```

Publish only after every chemistry box reached the interval boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/chemistry/device.py#L332-L345)
