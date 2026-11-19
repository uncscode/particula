# particula.gpu.dynamics.van_der_waals

*module*

Resident fp64 capture rates for compact polarizable spheres.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/van_der_waals.py#L1-L1)

## particula.gpu.dynamics.van_der_waals._Indices

*attribute* · *module attribute*

```python
_Indices = wp.types.vector(RADIAL_INTERVALS + 1, wp.int32)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/van_der_waals.py#L25-L25)

## particula.gpu.dynamics.van_der_waals._KB

*attribute* · *module attribute*

```python
_KB = wp.constant(wp.float64(BOLTZMANN_CONSTANT))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/van_der_waals.py#L23-L23)

## particula.gpu.dynamics.van_der_waals._LENGTH_T

*attribute* · *module attribute*

```python
_LENGTH_T = wp.constant(wp.float64(COULOMB_LENGTH_T))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/van_der_waals.py#L20-L20)

## particula.gpu.dynamics.van_der_waals._LOG_STEP

*attribute* · *module attribute*

```python
_LOG_STEP = wp.constant(wp.float64(LOG_RATIO_STEP))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/van_der_waals.py#L21-L21)

## particula.gpu.dynamics.van_der_waals._Profile

*attribute* · *module attribute*

```python
_Profile = wp.types.vector(RADIAL_INTERVALS + 1, wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/van_der_waals.py#L24-L24)

## particula.gpu.dynamics.van_der_waals._RATIO_LIMIT

*attribute* · *module attribute*

```python
_RATIO_LIMIT = wp.constant(wp.float64(MAX_RADIUS_RATIO))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/van_der_waals.py#L22-L22)

## particula.gpu.dynamics.van_der_waals.CaptureData

*class*

```python
class CaptureData
```

Immutable geometry and material for one prepared mechanism.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/van_der_waals.py#L28-L35)

### particula.gpu.dynamics.van_der_waals.CaptureData.hamaker

*attribute* · *instance attribute*

```python
hamaker: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/van_der_waals.py#L32-L32)

### particula.gpu.dynamics.van_der_waals.CaptureData.polarizable

*attribute* · *instance attribute*

```python
polarizable: wp.int32
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/van_der_waals.py#L33-L33)

### particula.gpu.dynamics.van_der_waals.CaptureData.response

*attribute* · *instance attribute*

```python
response: wp.array3d(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/van_der_waals.py#L35-L35)

### particula.gpu.dynamics.van_der_waals.CaptureData.x

*attribute* · *instance attribute*

```python
x: wp.array(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/van_der_waals.py#L34-L34)

## particula.gpu.dynamics.van_der_waals._expm1

*function*

```python
def _expm1(x: wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/van_der_waals.py#L52-L59)

## particula.gpu.dynamics.van_der_waals._hamaker

*function*

```python
def _hamaker(x: wp.float64, fraction: wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/van_der_waals.py#L62-L90)

## particula.gpu.dynamics.van_der_waals.capture_pair_wp

*function*

```python
def capture_pair_wp(data: CaptureData, radius_i: wp.float64, radius_j: wp.float64, mass_i: wp.float64, mass_j: wp.float64, z_i: wp.float64, z_j: wp.float64, diffusion_sum: wp.float64, temperature: wp.float64)
```

Evaluate the actual pair; negative return signals domain rejection.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/van_der_waals.py#L93-L232)

## particula.gpu.dynamics.van_der_waals.prepare_capture

*function*

```python
def prepare_capture(hamaker, epsilon, device)
```

Transfer the dielectric response once, outside native graph capture.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/van_der_waals.py#L38-L49)
