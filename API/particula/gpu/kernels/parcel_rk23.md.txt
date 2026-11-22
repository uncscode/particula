# particula.gpu.kernels.parcel_rk23

*module*

Shared adaptive warm-parcel integration and explicit RK3(2) stages.

The independent coordinates are p/p0 and liquid volume / dry volume per
class. At every stage total water and the parcel first law determine vapor
and temperature. Only scalar trial errors cross to the host during integration.
The accepted trajectory uses the selected third-order method on replay.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L1-L1)

## particula.gpu.kernels.parcel_rk23._BATCH_CAPACITY

*attribute* · *module attribute*

```python
_BATCH_CAPACITY = 32
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L39-L39)

## particula.gpu.kernels.parcel_rk23._Parcel

*class*

```python
class _Parcel
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L42-L51)

### particula.gpu.kernels.parcel_rk23._Parcel.dry_mass

*attribute* · *instance attribute*

```python
dry_mass: F
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L49-L49)

### particula.gpu.kernels.parcel_rk23._Parcel.exchange

*attribute* · *instance attribute*

```python
exchange: bool
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L51-L51)

### particula.gpu.kernels.parcel_rk23._Parcel.height

*attribute* · *instance attribute*

```python
height: F
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L50-L50)

### particula.gpu.kernels.parcel_rk23._Parcel.kappa

*attribute* · *instance attribute*

```python
kappa: F
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L48-L48)

### particula.gpu.kernels.parcel_rk23._Parcel.liquid

*attribute* · *instance attribute*

```python
liquid: F
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L47-L47)

### particula.gpu.kernels.parcel_rk23._Parcel.pressure

*attribute* · *instance attribute*

```python
pressure: F
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L45-L45)

### particula.gpu.kernels.parcel_rk23._Parcel.temperature

*attribute* · *instance attribute*

```python
temperature: F
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L44-L44)

### particula.gpu.kernels.parcel_rk23._Parcel.vapor

*attribute* · *instance attribute*

```python
vapor: F
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L46-L46)

## particula.gpu.kernels.parcel_rk23._ParcelStepper

*class*

```python
class _ParcelStepper
```

Shared physical rates and accepted-step reverse replay.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L271-L345)

### particula.gpu.kernels.parcel_rk23._ParcelStepper.device

*attribute* · *instance attribute*

```python
device = device
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L275-L275)

### particula.gpu.kernels.parcel_rk23._ParcelStepper.dry

*attribute* · *instance attribute*

```python
dry = self.array(initial.dry_radius)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L278-L278)

### particula.gpu.kernels.parcel_rk23._ParcelStepper.initial_state

*attribute* · *instance attribute*

```python
initial_state = self.array(np.r_[1.0, water])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L283-L283)

### particula.gpu.kernels.parcel_rk23._ParcelStepper.parcel

*attribute* · *instance attribute*

```python
parcel = _Parcel()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L285-L285)

### particula.gpu.kernels.parcel_rk23._ParcelStepper.seed

*attribute* · *instance attribute*

```python
seed = self.array(np.r_[0.0, weights]) if gradient else None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L284-L284)

### particula.gpu.kernels.parcel_rk23._ParcelStepper.speed

*attribute* · *instance attribute*

```python
speed = self.array([speed], grad=gradient)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L277-L277)

### particula.gpu.kernels.parcel_rk23._ParcelStepper.weights

*attribute* · *instance attribute*

```python
weights = self.array(weights)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L281-L281)

### particula.gpu.kernels.parcel_rk23._ParcelStepper.width

*attribute* · *instance attribute*

```python
width = len(wet) + 1
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L276-L276)

### particula.gpu.kernels.parcel_rk23._ParcelStepper.__init__

*method*

```python
def __init__(initial, wet, speed, device, gradient, exchange)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L274-L298)

### particula.gpu.kernels.parcel_rk23._ParcelStepper.advance

*method*

```python
def advance(state, t, h, out, work)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L320-L321)

### particula.gpu.kernels.parcel_rk23._ParcelStepper.array

*method*

```python
def array(values, grad=False)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L300-L301)

### particula.gpu.kernels.parcel_rk23._ParcelStepper.empty

*method*

```python
def empty(shape, grad=False)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L303-L304)

### particula.gpu.kernels.parcel_rk23._ParcelStepper.launch

*method*

```python
def launch(kernel, dim, args, out)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L306-L309)

### particula.gpu.kernels.parcel_rk23._ParcelStepper.rate

*method*

```python
def rate(state, t, derivative, liquid)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L311-L318)

### particula.gpu.kernels.parcel_rk23._ParcelStepper.record_step

*method* · *staticmethod*

```python
def record_step(checkpoints, accepted, start, t, h)
```

Keep one device checkpoint for each block of accepted intervals.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L326-L331)

### particula.gpu.kernels.parcel_rk23._ParcelStepper.reverse

*method*

```python
def reverse(checkpoints)
```

Bound tape storage by replaying at most 32 accepted steps at once.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L333-L345)

### particula.gpu.kernels.parcel_rk23._ParcelStepper.scratch

*method*

```python
def scratch(grad=False)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L323-L324)

## particula.gpu.kernels.parcel_rk23._RK23Stepper

*class*

```python
class _RK23Stepper(_ParcelStepper)
```

Bases: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L348-L370)

### particula.gpu.kernels.parcel_rk23._RK23Stepper.device

*attribute* · *instance attribute*

```python
device = device
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L275-L275)

### particula.gpu.kernels.parcel_rk23._RK23Stepper.dry

*attribute* · *instance attribute*

```python
dry = self.array(initial.dry_radius)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L278-L278)

### particula.gpu.kernels.parcel_rk23._RK23Stepper.initial_state

*attribute* · *instance attribute*

```python
initial_state = self.array(np.r_[1.0, water])
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L283-L283)

### particula.gpu.kernels.parcel_rk23._RK23Stepper.parcel

*attribute* · *instance attribute*

```python
parcel = _Parcel()
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L285-L285)

### particula.gpu.kernels.parcel_rk23._RK23Stepper.seed

*attribute* · *instance attribute*

```python
seed = self.array(np.r_[0.0, weights]) if gradient else None
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L284-L284)

### particula.gpu.kernels.parcel_rk23._RK23Stepper.speed

*attribute* · *instance attribute*

```python
speed = self.array([speed], grad=gradient)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L277-L277)

### particula.gpu.kernels.parcel_rk23._RK23Stepper.weights

*attribute* · *instance attribute*

```python
weights = self.array(weights)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L281-L281)

### particula.gpu.kernels.parcel_rk23._RK23Stepper.width

*attribute* · *instance attribute*

```python
width = len(wet) + 1
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L276-L276)

### particula.gpu.kernels.parcel_rk23._RK23Stepper.advance

*method*

```python
def advance(state, t, h, out, work, *, reuse_first=False)
```

Write third-order result, retaining distinct stage values for AD.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L349-L364)

### particula.gpu.kernels.parcel_rk23._RK23Stepper.scratch

*method*

```python
def scratch(grad=False)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L366-L370)

### particula.gpu.kernels.parcel_rk23._RK23Stepper.__init__

*method*

```python
def __init__(initial, wet, speed, device, gradient, exchange)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L274-L298)

### particula.gpu.kernels.parcel_rk23._RK23Stepper.array

*method*

```python
def array(values, grad=False)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L300-L301)

### particula.gpu.kernels.parcel_rk23._RK23Stepper.empty

*method*

```python
def empty(shape, grad=False)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L303-L304)

### particula.gpu.kernels.parcel_rk23._RK23Stepper.launch

*method*

```python
def launch(kernel, dim, args, out)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L306-L309)

### particula.gpu.kernels.parcel_rk23._RK23Stepper.rate

*method*

```python
def rate(state, t, derivative, liquid)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L311-L318)

### particula.gpu.kernels.parcel_rk23._RK23Stepper.record_step

*method* · *staticmethod*

```python
def record_step(checkpoints, accepted, start, t, h)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

Keep one device checkpoint for each block of accepted intervals.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L326-L331)

### particula.gpu.kernels.parcel_rk23._RK23Stepper.reverse

*method*

```python
def reverse(checkpoints)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

Bound tape storage by replaying at most 32 accepted steps at once.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L333-L345)

## particula.gpu.kernels.parcel_rk23._RK23Trials

*class*

```python
class _RK23Trials(_TrialBatch)
```

Bases: `particula.gpu.kernels.parcel_rk23._TrialBatch`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L425-L471)

### particula.gpu.kernels.parcel_rk23._RK23Trials.atol

*attribute* · *instance attribute*

```python
atol = solver.array(np.full(solver.width, tolerance * 1e-08))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L429-L429)

### particula.gpu.kernels.parcel_rk23._RK23Trials.k4

*attribute* · *instance attribute*

```python
k4 = solver.empty(solver.width)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L431-L431)

### particula.gpu.kernels.parcel_rk23._RK23Trials.ql

*attribute* · *instance attribute*

```python
ql = solver.empty(1)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L432-L432)

### particula.gpu.kernels.parcel_rk23._RK23Trials.rtol

*attribute* · *instance attribute*

```python
rtol = solver.array(np.full(solver.width, tolerance))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L428-L428)

### particula.gpu.kernels.parcel_rk23._RK23Trials.work

*attribute* · *instance attribute*

```python
work = solver.scratch()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L430-L430)

### particula.gpu.kernels.parcel_rk23._RK23Trials.capacity

*attribute* · *instance attribute*

```python
capacity = _BATCH_CAPACITY
```

Inherited from: `particula.gpu.kernels.parcel_rk23._TrialBatch`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L381-L381)

### particula.gpu.kernels.parcel_rk23._RK23Trials.errors

*attribute* · *instance attribute*

```python
errors = solver.empty((self.capacity, 1))
```

Inherited from: `particula.gpu.kernels.parcel_rk23._TrialBatch`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L383-L383)

### particula.gpu.kernels.parcel_rk23._RK23Trials.solver

*attribute* · *instance attribute*

```python
solver = solver
```

Inherited from: `particula.gpu.kernels.parcel_rk23._TrialBatch`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L377-L377)

### particula.gpu.kernels.parcel_rk23._RK23Trials.states

*attribute* · *instance attribute*

```python
states = [solver.empty(solver.width) for _ in range(self.capacity)]
```

Inherited from: `particula.gpu.kernels.parcel_rk23._TrialBatch`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L382-L382)

### particula.gpu.kernels.parcel_rk23._RK23Trials.tolerance

*attribute* · *instance attribute*

```python
tolerance = tolerance
```

Inherited from: `particula.gpu.kernels.parcel_rk23._TrialBatch`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L378-L378)

### particula.gpu.kernels.parcel_rk23._RK23Trials.__init__

*method*

```python
def __init__(solver, tolerance)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L426-L432)

### particula.gpu.kernels.parcel_rk23._RK23Trials.reuse_endpoint

*method*

```python
def reuse_endpoint()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L467-L471)

### particula.gpu.kernels.parcel_rk23._RK23Trials.start

*method*

```python
def start(state, t)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L434-L435)

### particula.gpu.kernels.parcel_rk23._RK23Trials.trial

*method*

```python
def trial(state, t, h, out, error)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L437-L465)

### particula.gpu.kernels.parcel_rk23._RK23Trials.evaluate

*method*

```python
def evaluate(initial, start, target, proposed, count)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._TrialBatch`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L385-L413)

## particula.gpu.kernels.parcel_rk23._Stages

*class* · *dataclass*

```python
class _Stages
```

Three slope vectors, two stage states and their saved water sums.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L257-L268)

### particula.gpu.kernels.parcel_rk23._Stages.k1

*attribute* · *instance attribute*

```python
k1: FloatVector
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L261-L261)

### particula.gpu.kernels.parcel_rk23._Stages.k2

*attribute* · *instance attribute*

```python
k2: FloatVector
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L262-L262)

### particula.gpu.kernels.parcel_rk23._Stages.k3

*attribute* · *instance attribute*

```python
k3: FloatVector
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L263-L263)

### particula.gpu.kernels.parcel_rk23._Stages.q1

*attribute* · *instance attribute*

```python
q1: FloatVector
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L266-L266)

### particula.gpu.kernels.parcel_rk23._Stages.q2

*attribute* · *instance attribute*

```python
q2: FloatVector
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L267-L267)

### particula.gpu.kernels.parcel_rk23._Stages.q3

*attribute* · *instance attribute*

```python
q3: FloatVector
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L268-L268)

### particula.gpu.kernels.parcel_rk23._Stages.s2

*attribute* · *instance attribute*

```python
s2: FloatVector
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L264-L264)

### particula.gpu.kernels.parcel_rk23._Stages.s3

*attribute* · *instance attribute*

```python
s3: FloatVector
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L265-L265)

### particula.gpu.kernels.parcel_rk23._Stages.__init__

*method*

```python
def __init__(k1: FloatVector, k2: FloatVector, k3: FloatVector, s2: FloatVector, s3: FloatVector, q1: FloatVector, q2: FloatVector, q3: FloatVector) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L0-L0)

## particula.gpu.kernels.parcel_rk23._TrialBatch

*class*

```python
class _TrialBatch
```

Stage speculative candidates in disjoint rows until the next decision.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L373-L422)

### particula.gpu.kernels.parcel_rk23._TrialBatch.capacity

*attribute* · *instance attribute*

```python
capacity = _BATCH_CAPACITY
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L381-L381)

### particula.gpu.kernels.parcel_rk23._TrialBatch.errors

*attribute* · *instance attribute*

```python
errors = solver.empty((self.capacity, 1))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L383-L383)

### particula.gpu.kernels.parcel_rk23._TrialBatch.solver

*attribute* · *instance attribute*

```python
solver = solver
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L377-L377)

### particula.gpu.kernels.parcel_rk23._TrialBatch.states

*attribute* · *instance attribute*

```python
states = [solver.empty(solver.width) for _ in range(self.capacity)]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L382-L382)

### particula.gpu.kernels.parcel_rk23._TrialBatch.tolerance

*attribute* · *instance attribute*

```python
tolerance = tolerance
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L378-L378)

### particula.gpu.kernels.parcel_rk23._TrialBatch.__init__

*method*

```python
def __init__(solver, tolerance)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L376-L383)

### particula.gpu.kernels.parcel_rk23._TrialBatch.evaluate

*method*

```python
def evaluate(initial, start, target, proposed, count)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L385-L413)

### particula.gpu.kernels.parcel_rk23._TrialBatch.reuse_endpoint

*method*

```python
def reuse_endpoint()
```

Carry endpoint derivatives into the next candidate if available.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L421-L422)

### particula.gpu.kernels.parcel_rk23._TrialBatch.start

*method*

```python
def start(state, t)
```

Prepare any reusable first-stage derivative.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L415-L416)

### particula.gpu.kernels.parcel_rk23._TrialBatch.trial

*method*

```python
def trial(state, t, h, out, error)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L418-L419)

## particula.gpu.kernels.parcel_rk23._adaptive_method

*function*

```python
def _adaptive_method(controls)
```

Select the concrete Warp stepper and its error estimate.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L474-L480)

## particula.gpu.kernels.parcel_rk23._rate

*function*

```python
def _rate(state: FloatVector, liquid: FloatVector, dry: FloatVector, speed: FloatVector, parcel: _Parcel, time: F, result: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L71-L124)

## particula.gpu.kernels.parcel_rk23._relaxation_rate

*function*

```python
def _relaxation_rate(state: FloatVector, dry: FloatVector, weights: FloatVector, vapor: F, temp: F, pressure: F, kappa: F)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L127-L162)

## particula.gpu.kernels.parcel_rk23._reservoir_error

*function*

```python
def _reservoir_error(state: FloatVector, dry: FloatVector, weights: FloatVector, k1: FloatVector, k2: FloatVector, k3: FloatVector, k4: FloatVector, old_liquid: FloatVector, new_liquid: FloatVector, speed: FloatVector, parcel: _Parcel, time: F, h: F, rtol: F, norm: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L165-L225)

## particula.gpu.kernels.parcel_rk23._sample

*function*

```python
def _sample(state: FloatVector, liquid: FloatVector, dry: FloatVector, speed: FloatVector, parcel: _Parcel, time: F, air: FloatVector, wet: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L228-L254)

## particula.gpu.kernels.parcel_rk23._sum_water

*function*

```python
def _sum_water(state: FloatVector, weights: FloatVector, liquid: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L62-L68)

## particula.gpu.kernels.parcel_rk23._temperature

*function*

```python
def _temperature(parcel: _Parcel, liquid: F, dz: F)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L54-L59)

## particula.gpu.kernels.parcel_rk23.run_adaptive_parcel

*function*

```python
def run_adaptive_parcel(initial, wet, time, updraft, device, gradient, phase_exchange, controls)
```

Integrate to fixed observations with an embedded componentwise error.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L483-L567)
