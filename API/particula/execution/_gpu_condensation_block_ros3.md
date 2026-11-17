# particula.execution._gpu_condensation_block_ros3

*module*

ROS3 stages with coupled material blocks and conservative reservoirs.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_block_ros3.py#L1-L1)

## particula.execution._gpu_condensation_block_ros3.BlockROS3Stages

*class*

```python
class BlockROS3Stages(ROS3Stages)
```

Bases: `particula.execution._gpu_condensation_ros3.ROS3Stages`

Store O(particles * exchanging_materials**2) device scratch.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_block_ros3.py#L13-L225)

### particula.execution._gpu_condensation_block_ros3.BlockROS3Stages._factorized

*attribute* · *instance attribute*

```python
_factorized = False
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_block_ros3.py#L55-L55)

### particula.execution._gpu_condensation_block_ros3.BlockROS3Stages.augmented

*attribute* · *instance attribute*

```python
augmented = owner._array(np.zeros((particles, self.width, 2 * self.width)))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_block_ros3.py#L30-L32)

### particula.execution._gpu_condensation_block_ros3.BlockROS3Stages.h

*attribute* · *instance attribute*

```python
h = owner._array(np.ones(1))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_block_ros3.py#L54-L54)

### particula.execution._gpu_condensation_block_ros3.BlockROS3Stages.increments

*attribute* · *instance attribute*

```python
increments = tuple((owner._array(np.zeros(owner._size)) for _ in range(3)))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_block_ros3.py#L22-L24)

### particula.execution._gpu_condensation_block_ros3.BlockROS3Stages.pairs

*attribute* · *instance attribute*

```python
pairs = pairs
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_block_ros3.py#L17-L17)

### particula.execution._gpu_condensation_block_ros3.BlockROS3Stages.particular

*attribute* · *instance attribute*

```python
particular = owner._array(np.zeros((particles, self.width)))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_block_ros3.py#L33-L33)

### particula.execution._gpu_condensation_block_ros3.BlockROS3Stages.parts

*attribute* · *instance attribute*

```python
parts = owner._array(np.zeros((owner._shape[0] * self.width, self.width + 1, max(1, min(64, owner._shape[1])))))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_block_ros3.py#L37-L45)

### particula.execution._gpu_condensation_block_ros3.BlockROS3Stages.reverse_source

*attribute* · *instance attribute*

```python
reverse_source = owner._array(reverse_source, wp.int32)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_block_ros3.py#L53-L53)

### particula.execution._gpu_condensation_block_ros3.BlockROS3Stages.reverse_target

*attribute* · *instance attribute*

```python
reverse_target = owner._array(reverse_target, wp.int32)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_block_ros3.py#L52-L52)

### particula.execution._gpu_condensation_block_ros3.BlockROS3Stages.schur

*attribute* · *instance attribute*

```python
schur = owner._array(np.zeros((owner._shape[0], self.width, self.width + 1)))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_block_ros3.py#L34-L36)

### particula.execution._gpu_condensation_block_ros3.BlockROS3Stages.sources

*attribute* · *instance attribute*

```python
sources = owner._array([s for (s, _) in pairs], wp.int32)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_block_ros3.py#L46-L46)

### particula.execution._gpu_condensation_block_ros3.BlockROS3Stages.targets

*attribute* · *instance attribute*

```python
targets = owner._array([t for (_, t) in pairs], wp.int32)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_block_ros3.py#L47-L47)

### particula.execution._gpu_condensation_block_ros3.BlockROS3Stages.width

*attribute* · *instance attribute*

```python
width = len(pairs)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_block_ros3.py#L18-L18)

### particula.execution._gpu_condensation_block_ros3.BlockROS3Stages.__init__

*method*

```python
def __init__(owner, pairs)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_block_ros3.py#L16-L55)

### particula.execution._gpu_condensation_block_ros3.BlockROS3Stages.linearize

*method*

```python
def linearize(owner)
```

Color independent particles, retaining every cross-material row.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_block_ros3.py#L57-L121)

### particula.execution._gpu_condensation_block_ros3.BlockROS3Stages.solve

*method*

```python
def solve(owner, h, result)
```

Factor once per trial and solve each stage on device.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_block_ros3.py#L123-L225)

### particula.execution._gpu_condensation_block_ros3.BlockROS3Stages.trial

*method*

```python
def trial(owner, h)
```

Inherited from: `particula.execution._gpu_condensation_ros3.ROS3Stages`

Check the full candidate and return the embedded maximum error.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_gpu_condensation_ros3.py#L137-L188)
