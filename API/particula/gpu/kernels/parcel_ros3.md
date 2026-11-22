# particula.gpu.kernels.parcel_ros3

*module*

Linearly implicit ROS3(2) for the closed warm parcel.

In normalized coordinates each row depends on local water, pressure and the
weighted sum of class water. Eliminating pressure and solving for that sum
gives an O(classes) solve for each of the three Rosenbrock increments. Warp
differentiates the factorization as well as the stages on accepted-step replay.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L1-L1)

## particula.gpu.kernels.parcel_ros3._ROS3Stages

*class* · *dataclass*

```python
class _ROS3Stages
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L230-L243)

### particula.gpu.kernels.parcel_ros3._ROS3Stages.base_rate

*attribute* · *instance attribute*

```python
base_rate: FloatVector
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L236-L236)

### particula.gpu.kernels.parcel_ros3._ROS3Stages.dl

*attribute* · *instance attribute*

```python
dl: FloatVector
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L234-L234)

### particula.gpu.kernels.parcel_ros3._ROS3Stages.dp

*attribute* · *instance attribute*

```python
dp: FloatVector
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L233-L233)

### particula.gpu.kernels.parcel_ros3._ROS3Stages.ft

*attribute* · *instance attribute*

```python
ft: FloatVector
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L235-L235)

### particula.gpu.kernels.parcel_ros3._ROS3Stages.increments

*attribute* · *instance attribute*

```python
increments: list
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L243-L243)

### particula.gpu.kernels.parcel_ros3._ROS3Stages.inverse

*attribute* · *instance attribute*

```python
inverse: FloatVector
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L232-L232)

### particula.gpu.kernels.parcel_ros3._ROS3Stages.old_liquid

*attribute* · *instance attribute*

```python
old_liquid: FloatVector
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L239-L239)

### particula.gpu.kernels.parcel_ros3._ROS3Stages.rhs

*attribute* · *instance attribute*

```python
rhs: list
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L241-L241)

### particula.gpu.kernels.parcel_ros3._ROS3Stages.stage

*attribute* · *instance attribute*

```python
stage: FloatVector
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L238-L238)

### particula.gpu.kernels.parcel_ros3._ROS3Stages.stage_liquid

*attribute* · *instance attribute*

```python
stage_liquid: FloatVector
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L240-L240)

### particula.gpu.kernels.parcel_ros3._ROS3Stages.stage_rate

*attribute* · *instance attribute*

```python
stage_rate: FloatVector
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L237-L237)

### particula.gpu.kernels.parcel_ros3._ROS3Stages.water

*attribute* · *instance attribute*

```python
water: list
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L242-L242)

### particula.gpu.kernels.parcel_ros3._ROS3Stages.__init__

*method*

```python
def __init__(inverse: FloatVector, dp: FloatVector, dl: FloatVector, ft: FloatVector, base_rate: FloatVector, stage_rate: FloatVector, stage: FloatVector, old_liquid: FloatVector, stage_liquid: FloatVector, rhs: list, water: list, increments: list) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L0-L0)

## particula.gpu.kernels.parcel_ros3._ROS3Stepper

*class*

```python
class _ROS3Stepper(_ParcelStepper)
```

Bases: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

Three solves with one parcel linearization per trial or replay.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L246-L316)

### particula.gpu.kernels.parcel_ros3._ROS3Stepper.device

*attribute* · *instance attribute*

```python
device = device
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L275-L275)

### particula.gpu.kernels.parcel_ros3._ROS3Stepper.dry

*attribute* · *instance attribute*

```python
dry = self.array(initial.dry_radius)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L278-L278)

### particula.gpu.kernels.parcel_ros3._ROS3Stepper.initial_state

*attribute* · *instance attribute*

```python
initial_state = self.array(np.r_[1.0, water])
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L283-L283)

### particula.gpu.kernels.parcel_ros3._ROS3Stepper.parcel

*attribute* · *instance attribute*

```python
parcel = _Parcel()
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L285-L285)

### particula.gpu.kernels.parcel_ros3._ROS3Stepper.seed

*attribute* · *instance attribute*

```python
seed = self.array(np.r_[0.0, weights]) if gradient else None
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L284-L284)

### particula.gpu.kernels.parcel_ros3._ROS3Stepper.speed

*attribute* · *instance attribute*

```python
speed = self.array([speed], grad=gradient)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L277-L277)

### particula.gpu.kernels.parcel_ros3._ROS3Stepper.weights

*attribute* · *instance attribute*

```python
weights = self.array(weights)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L281-L281)

### particula.gpu.kernels.parcel_ros3._ROS3Stepper.width

*attribute* · *instance attribute*

```python
width = len(wet) + 1
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L276-L276)

### particula.gpu.kernels.parcel_ros3._ROS3Stepper.advance

*method*

```python
def advance(state, t, h, out, work)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L259-L316)

### particula.gpu.kernels.parcel_ros3._ROS3Stepper.scratch

*method*

```python
def scratch(grad=False)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L249-L257)

### particula.gpu.kernels.parcel_ros3._ROS3Stepper.__init__

*method*

```python
def __init__(initial, wet, speed, device, gradient, exchange)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L274-L298)

### particula.gpu.kernels.parcel_ros3._ROS3Stepper.array

*method*

```python
def array(values, grad=False)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L300-L301)

### particula.gpu.kernels.parcel_ros3._ROS3Stepper.empty

*method*

```python
def empty(shape, grad=False)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L303-L304)

### particula.gpu.kernels.parcel_ros3._ROS3Stepper.launch

*method*

```python
def launch(kernel, dim, args, out)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L306-L309)

### particula.gpu.kernels.parcel_ros3._ROS3Stepper.rate

*method*

```python
def rate(state, t, derivative, liquid)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L311-L318)

### particula.gpu.kernels.parcel_ros3._ROS3Stepper.record_step

*method* · *staticmethod*

```python
def record_step(checkpoints, accepted, start, t, h)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

Keep one device checkpoint for each block of accepted intervals.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L326-L331)

### particula.gpu.kernels.parcel_ros3._ROS3Stepper.reverse

*method*

```python
def reverse(checkpoints)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._ParcelStepper`

Bound tape storage by replaying at most 32 accepted steps at once.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L333-L345)

## particula.gpu.kernels.parcel_ros3._ROS3Trials

*class*

```python
class _ROS3Trials(_TrialBatch)
```

Bases: `particula.gpu.kernels.parcel_rk23._TrialBatch`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L319-L355)

### particula.gpu.kernels.parcel_ros3._ROS3Trials.embedded

*attribute* · *instance attribute*

```python
embedded = solver.empty(solver.width)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L323-L323)

### particula.gpu.kernels.parcel_ros3._ROS3Trials.error_water

*attribute* · *instance attribute*

```python
error_water = solver.empty(1)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L324-L324)

### particula.gpu.kernels.parcel_ros3._ROS3Trials.new_water

*attribute* · *instance attribute*

```python
new_water = solver.empty(1)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L325-L325)

### particula.gpu.kernels.parcel_ros3._ROS3Trials.work

*attribute* · *instance attribute*

```python
work = solver.scratch()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L322-L322)

### particula.gpu.kernels.parcel_ros3._ROS3Trials.capacity

*attribute* · *instance attribute*

```python
capacity = _BATCH_CAPACITY
```

Inherited from: `particula.gpu.kernels.parcel_rk23._TrialBatch`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L381-L381)

### particula.gpu.kernels.parcel_ros3._ROS3Trials.errors

*attribute* · *instance attribute*

```python
errors = solver.empty((self.capacity, 1))
```

Inherited from: `particula.gpu.kernels.parcel_rk23._TrialBatch`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L383-L383)

### particula.gpu.kernels.parcel_ros3._ROS3Trials.solver

*attribute* · *instance attribute*

```python
solver = solver
```

Inherited from: `particula.gpu.kernels.parcel_rk23._TrialBatch`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L377-L377)

### particula.gpu.kernels.parcel_ros3._ROS3Trials.states

*attribute* · *instance attribute*

```python
states = [solver.empty(solver.width) for _ in range(self.capacity)]
```

Inherited from: `particula.gpu.kernels.parcel_rk23._TrialBatch`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L382-L382)

### particula.gpu.kernels.parcel_ros3._ROS3Trials.tolerance

*attribute* · *instance attribute*

```python
tolerance = tolerance
```

Inherited from: `particula.gpu.kernels.parcel_rk23._TrialBatch`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L378-L378)

### particula.gpu.kernels.parcel_ros3._ROS3Trials.__init__

*method*

```python
def __init__(solver, tolerance)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L320-L325)

### particula.gpu.kernels.parcel_ros3._ROS3Trials.trial

*method*

```python
def trial(state, t, h, out, error)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L327-L355)

### particula.gpu.kernels.parcel_ros3._ROS3Trials.evaluate

*method*

```python
def evaluate(initial, start, target, proposed, count)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._TrialBatch`

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L385-L413)

### particula.gpu.kernels.parcel_ros3._ROS3Trials.reuse_endpoint

*method*

```python
def reuse_endpoint()
```

Inherited from: `particula.gpu.kernels.parcel_rk23._TrialBatch`

Carry endpoint derivatives into the next candidate if available.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L421-L422)

### particula.gpu.kernels.parcel_ros3._ROS3Trials.start

*method*

```python
def start(state, t)
```

Inherited from: `particula.gpu.kernels.parcel_rk23._TrialBatch`

Prepare any reusable first-stage derivative.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_rk23.py#L415-L416)

## particula.gpu.kernels.parcel_ros3._back_substitute

*function*

```python
def _back_substitute(rhs: FloatVector, inv: FloatVector, dp: FloatVector, dl: FloatVector, liquid: FloatVector, result: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L144-L156)

## particula.gpu.kernels.parcel_ros3._error

*function*

```python
def _error(state: FloatVector, candidate: FloatVector, first: FloatVector, second: FloatVector, third: FloatVector, rtol: F, embedded: FloatVector, norm: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L171-L195)

## particula.gpu.kernels.parcel_ros3._finish

*function*

```python
def _finish(state: FloatVector, first: FloatVector, second: FloatVector, third: FloatVector, out: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L159-L168)

## particula.gpu.kernels.parcel_ros3._linearize

*function*

```python
def _linearize(state: FloatVector, liquid: FloatVector, rate: FloatVector, dry: FloatVector, speed: FloatVector, parcel: _Parcel, time: F, h: F, inverse_diagonal: FloatVector, pressure_column: FloatVector, liquid_column: FloatVector, partial_time: FloatVector)
```

Form the local, pressure, shared-water and explicit time derivatives.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L42-L119)

## particula.gpu.kernels.parcel_ros3._reservoir_norm

*function*

```python
def _reservoir_norm(old: FloatVector, new_liquid: FloatVector, error: FloatVector, speed: FloatVector, parcel: _Parcel, t: F, h: F, rtol: F, norm: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L198-L227)

## particula.gpu.kernels.parcel_ros3._water_increment

*function*

```python
def _water_increment(rhs: FloatVector, inv: FloatVector, dp: FloatVector, dl: FloatVector, weights: FloatVector, result: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_ros3.py#L122-L141)
