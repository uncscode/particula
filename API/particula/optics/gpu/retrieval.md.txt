# particula.optics.gpu.retrieval

*module*

Warp parameter transforms for optical retrieval.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L1-L1)

## particula.optics.gpu.retrieval._NEG_THIRD

*attribute* · *module attribute*

```python
_NEG_THIRD = wp.constant(wp.float64(-1 / 3))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L17-L17)

## particula.optics.gpu.retrieval._ONE

*attribute* · *module attribute*

```python
_ONE = wp.constant(wp.float64(1))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L14-L14)

## particula.optics.gpu.retrieval._THIRD

*attribute* · *module attribute*

```python
_THIRD = wp.constant(wp.float64(1 / 3))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L16-L16)

## particula.optics.gpu.retrieval._TWO

*attribute* · *module attribute*

```python
_TWO = wp.constant(wp.float64(2))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L15-L15)

## particula.optics.gpu.retrieval._ZERO

*attribute* · *module attribute*

```python
_ZERO = wp.constant(wp.float64(0))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L13-L13)

## particula.optics.gpu.retrieval.DeviceRetrieval

*class*

```python
class DeviceRetrieval
```

Bind a small-parameter optical retrieval to an explicit Warp device.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L132-L225)

### particula.optics.gpu.retrieval.DeviceRetrieval._bulk

*attribute* · *instance attribute*

```python
_bulk = BulkOpticsAdjoint(shape, wavelength, device=device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L137-L137)

### particula.optics.gpu.retrieval.DeviceRetrieval._density

*attribute* · *instance attribute*

```python
_density = self._array(model.grid.number_density)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L157-L157)

### particula.optics.gpu.retrieval.DeviceRetrieval._device

*attribute* · *instance attribute*

```python
_device = self._bulk.device
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L139-L139)

### particula.optics.gpu.retrieval.DeviceRetrieval._effective

*attribute* · *instance attribute*

```python
_effective = isinstance(model, EffectiveIndexModel)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L140-L140)

### particula.optics.gpu.retrieval.DeviceRetrieval._index

*attribute* · *instance attribute*

```python
_index = self._complex_array(model._index)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L165-L165)

### particula.optics.gpu.retrieval.DeviceRetrieval._inputs

*attribute* · *instance attribute*

```python
_inputs = (self._array(model.grid.radius), self._array(rh), self._array(model._sizing_rh), self._array(model.kelvin), self._complex_array(lorentz_lorenz(model.water_index)), self._array(ratio), wp.float64(model._kappa or 0), int(len(model.parameters) == 3), int(model._rh is not None or model._ratio is not None), int(model._ratio is not None))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L145-L156)

### particula.optics.gpu.retrieval.DeviceRetrieval._radii

*attribute* · *instance attribute*

```python
_radii = self._array(model._radius)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L164-L164)

### particula.optics.gpu.retrieval.DeviceRetrieval._shape

*attribute* · *instance attribute*

```python
_shape = shape
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L138-L138)

### particula.optics.gpu.retrieval.DeviceRetrieval.__init__

*method*

```python
def __init__(model, shape, wavelength, *, device)
```

Upload coordinates fixed across all evaluations.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L135-L165)

### particula.optics.gpu.retrieval.DeviceRetrieval._array

*method*

```python
def _array(value, *, dtype=wp.float64, requires_grad=False)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L167-L173)

### particula.optics.gpu.retrieval.DeviceRetrieval._complex_array

*method*

```python
def _complex_array(value)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L175-L179)

### particula.optics.gpu.retrieval.DeviceRetrieval._population

*method*

```python
def _population(p, grad)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L181-L203)

### particula.optics.gpu.retrieval.DeviceRetrieval.evaluate

*method*

```python
def evaluate(parameters, seeds)
```

Return predictions and optional projected Jacobian rows.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L205-L225)

## particula.optics.gpu.retrieval._effective_population

*function*

```python
def _effective_population(p: wp.array(dtype=wp.float64), dry: wp.array2d(dtype=wp.float64), relative_humidity: wp.array(dtype=wp.float64), sizing_rh: wp.array(dtype=wp.float64), kelvin: wp.array(dtype=wp.float64), water_polar: wp.array(dtype=wp.vec2d), water_ratio: wp.array2d(dtype=wp.float64), fixed_kappa: wp.float64, fitted_kappa: int, humidify: int, measured_growth: int, radius: wp.array2d(dtype=wp.float64), index: wp.array2d(dtype=wp.vec2d))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L69-L111)

## particula.optics.gpu.retrieval._partition_weights

*function*

```python
def _partition_weights(p: wp.array(dtype=wp.float64), band: wp.array2d(dtype=wp.int32), endpoint: wp.array2d(dtype=wp.float64), subpopulations: int, weight: wp.array2d(dtype=wp.float64))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L114-L129)

## particula.optics.gpu.retrieval._refractive_index

*function*

```python
def _refractive_index(polar: wp.vec2d)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L57-L66)

## particula.optics.gpu.retrieval._water_ratio

*function*

```python
def _water_ratio(kappa: wp.float64, rh: wp.float64, bend: wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L20-L32)

## particula.optics.gpu.retrieval._water_ratio_grad

*function*

```python
def _water_ratio_grad(kappa: wp.float64, rh: wp.float64, bend: wp.float64, adj: wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/retrieval.py#L35-L54)
