# particula.gpu.kernels.activity_guard

*module*

Propagate neutral activity intervals through the configured box rate law.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/activity_guard.py#L1-L1)

## particula.gpu.kernels.activity_guard.FloatTensor

*attribute* · *module attribute*

```python
FloatTensor = Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/activity_guard.py#L8-L8)

## particula.gpu.kernels.activity_guard.IntMatrix

*attribute* · *module attribute*

```python
IntMatrix = Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/activity_guard.py#L9-L9)

## particula.gpu.kernels.activity_guard.BoxFluxGuard

*class*

```python
class BoxFluxGuard
```

Use the configured box rate law at both activity endpoints.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/activity_guard.py#L49-L77)

### particula.gpu.kernels.activity_guard.BoxFluxGuard.binding

*attribute* · *instance attribute*

```python
binding = binding
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/activity_guard.py#L54-L54)

### particula.gpu.kernels.activity_guard.BoxFluxGuard.bound

*attribute* · *instance attribute*

```python
bound = wp.empty_like(binding.values)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/activity_guard.py#L55-L55)

### particula.gpu.kernels.activity_guard.BoxFluxGuard.flat

*attribute* · *instance attribute*

```python
flat = self.bound.flatten()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/activity_guard.py#L56-L56)

### particula.gpu.kernels.activity_guard.BoxFluxGuard.high

*attribute* · *instance attribute*

```python
high = wp.empty_like(self.bound)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/activity_guard.py#L58-L58)

### particula.gpu.kernels.activity_guard.BoxFluxGuard.low

*attribute* · *instance attribute*

```python
low = wp.empty_like(self.bound)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/activity_guard.py#L57-L57)

### particula.gpu.kernels.activity_guard.BoxFluxGuard.__init__

*method*

```python
def __init__(binding)
```

Allocate interval and proposal buffers before capture.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/activity_guard.py#L52-L58)

### particula.gpu.kernels.activity_guard.BoxFluxGuard.refine

*method*

```python
def refine(rate, masses, counts, temperature, invalid, stage)
```

Refresh all targets for each row whose rate interval spans zero.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/activity_guard.py#L60-L77)

## particula.gpu.kernels.activity_guard._bound

*function*

```python
def _bound(activity: FloatTensor, error: FloatTensor, sign: wp.float64, out: FloatTensor)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/activity_guard.py#L15-L23)

## particula.gpu.kernels.activity_guard._crosses

*function*

```python
def _crosses(low: FloatTensor, high: FloatTensor, error: FloatTensor, mask: IntMatrix)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/activity_guard.py#L26-L46)
