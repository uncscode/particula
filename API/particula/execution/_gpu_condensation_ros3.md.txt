# particula.execution._gpu_condensation_ros3

*module*

Resident ROS3 stages for the single-exchange condensation profile.

The accepted state and lifecycle belong to GPUCondensationODE. This helper
owns only trial scratch and never downloads a physical state. Derivatives use
native device rates; a conservative arrowhead solve replaces dense LU storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_ros3.py#L1-L1)

## particula.execution._gpu_condensation_ros3.ROS3Stages

*class*

```python
class ROS3Stages
```

Allocate linear-storage scratch on the owner's requested device.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_ros3.py#L15-L188)

### particula.execution._gpu_condensation_ros3.ROS3Stages.increments

*attribute* · *instance attribute*

```python
increments = tuple((owner._array(np.zeros(owner._size)) for _ in range(3)))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_ros3.py#L23-L25)

### particula.execution._gpu_condensation_ros3.ROS3Stages.__init__

*method*

```python
def __init__(owner, pair)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_ros3.py#L18-L28)

### particula.execution._gpu_condensation_ros3.ROS3Stages.linearize

*method*

```python
def linearize(owner)
```

Differentiate local particle/gas columns at the accepted state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_ros3.py#L30-L89)

### particula.execution._gpu_condensation_ros3.ROS3Stages.solve

*method*

```python
def solve(owner, h, result)
```

Eliminate gas coupling and back-substitute particle increments.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_ros3.py#L91-L135)

### particula.execution._gpu_condensation_ros3.ROS3Stages.trial

*method*

```python
def trial(owner, h)
```

Check the full candidate and return the embedded maximum error.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_ros3.py#L137-L188)
