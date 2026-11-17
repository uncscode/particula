# particula.execution._condensation_device_control

*module*

GPU-owned adaptive clock, acceptance and step-size control.

The host supplies an observation bound. A native CUDA conditional graph runs
trials to that bound without control readbacks. Float control entries are
h, time, next_step, bound, min_step, max_step, last_error. Integer counters
are accepted, rejected, domain_rejected, failure_code, consecutive_rejections.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_device_control.py#L1-L1)

## particula.execution._condensation_device_control.accumulate_invalid

*function*

```python
def accumulate_invalid(current: IntVector, sticky: IntVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_device_control.py#L44-L47)

## particula.execution._condensation_device_control.begin_trial

*function*

```python
def begin_trial(control: FloatVector, invalid: IntVector, norm: FloatVector, accepted: IntVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_device_control.py#L31-L41)

## particula.execution._condensation_device_control.commit_candidate

*function*

```python
def commit_candidate(state: FloatVector, candidate: FloatVector, accepted: IntVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_device_control.py#L108-L114)

## particula.execution._condensation_device_control.decide

*function*

```python
def decide(control: FloatVector, norm: FloatVector, invalid: IntVector, accepted: IntVector, condition: IntVector, counters: IntVector, max_trials: int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_device_control.py#L50-L105)

## particula.execution._condensation_device_control.import_physical

*function*

```python
def import_physical(masses: FloatTensor, gas: FloatMatrix, physical_atol: FloatVector, state: FloatVector, scale: FloatVector, initial_scaled: FloatVector, initial_physical: FloatVector, atol: FloatVector)
```

Reset scaled coordinates at a split boundary without leaving device.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_device_control.py#L133-L165)

## particula.execution._condensation_device_control.reset_clock

*function*

```python
def reset_clock(control: FloatVector, counters: IntVector, time: wp.float64, first_step: wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_device_control.py#L117-L130)

## particula.execution._condensation_device_control.rk_error

*function*

```python
def rk_error(state: FloatVector, candidate: FloatVector, k1: FloatVector, k2: FloatVector, k3: FloatVector, k4: FloatVector, atol: FloatVector, rtol: FloatVector, control: FloatVector, norm: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_device_control.py#L191-L216)

## particula.execution._condensation_device_control.rk_stage

*function*

```python
def rk_stage(state: FloatVector, k1: FloatVector, k2: FloatVector, k3: FloatVector, control: FloatVector, a: wp.float64, b: wp.float64, c: wp.float64, output: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_device_control.py#L168-L188)

## particula.execution._condensation_device_control.ros_rhs

*function*

```python
def ros_rhs(rate: FloatVector, first: FloatVector, second: FloatVector, control: FloatVector, stage: int, rhs: FloatVector)
```

Autonomous ROS3 stage RHS; environment is fixed inside each segment.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_device_control.py#L219-L235)

## particula.execution._condensation_device_control.set_bound

*function*

```python
def set_bound(control: FloatVector, condition: IntVector, counters: IntVector, bound: wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_device_control.py#L20-L28)
