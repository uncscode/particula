# particula.optics.gpu._shortwave

*module*

Device discrete ordinates and layer adding for scalar shortwave transport.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L1-L1)

## particula.optics.gpu._shortwave.F

*attribute* · *module attribute*

```python
F = wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L10-L10)

## particula.optics.gpu._shortwave._PI

*attribute* · *module attribute*

```python
_PI = wp.constant(F(pi))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L11-L11)

## particula.optics.gpu._shortwave.BoundaryData

*class*

```python
class BoundaryData
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L21-L33)

### particula.optics.gpu._shortwave.BoundaryData.basis

*attribute* · *instance attribute*

```python
basis: wp.array4d(dtype=F)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L25-L25)

### particula.optics.gpu._shortwave.BoundaryData.diffuse

*attribute* · *instance attribute*

```python
diffuse: wp.array2d(dtype=F)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L32-L32)

### particula.optics.gpu._shortwave.BoundaryData.direct

*attribute* · *instance attribute*

```python
direct: wp.array2d(dtype=F)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L31-L31)

### particula.optics.gpu._shortwave.BoundaryData.direct_surface

*attribute* · *instance attribute*

```python
direct_surface: wp.array4d(dtype=F)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L27-L27)

### particula.optics.gpu._shortwave.BoundaryData.identity

*attribute* · *instance attribute*

```python
identity: wp.array(dtype=int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L23-L23)

### particula.optics.gpu._shortwave.BoundaryData.leaving

*attribute* · *instance attribute*

```python
leaving: wp.array2d(dtype=F)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L33-L33)

### particula.optics.gpu._shortwave.BoundaryData.mu

*attribute* · *instance attribute*

```python
mu: wp.array(dtype=F)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L28-L28)

### particula.optics.gpu._shortwave.BoundaryData.mu0

*attribute* · *instance attribute*

```python
mu0: wp.array(dtype=F)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L30-L30)

### particula.optics.gpu._shortwave.BoundaryData.reflection

*attribute* · *instance attribute*

```python
reflection: wp.array4d(dtype=F)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L26-L26)

### particula.optics.gpu._shortwave.BoundaryData.weight

*attribute* · *instance attribute*

```python
weight: wp.array(dtype=F)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L29-L29)

## particula.optics.gpu._shortwave.LayerData

*class*

```python
class LayerData
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L14-L18)

### particula.optics.gpu._shortwave.LayerData.albedo

*attribute* · *instance attribute*

```python
albedo: wp.array3d(dtype=F)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L17-L17)

### particula.optics.gpu._shortwave.LayerData.depth

*attribute* · *instance attribute*

```python
depth: wp.array3d(dtype=F)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L16-L16)

### particula.optics.gpu._shortwave.LayerData.moments

*attribute* · *instance attribute*

```python
moments: wp.array4d(dtype=F)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L18-L18)

## particula.optics.gpu._shortwave._finite_flux

*function*

```python
def _finite_flux(value: wp.vec3d)
```

Check all three directional flux components for NaN or infinity.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L325-L332)

## particula.optics.gpu._shortwave._scaled_depth

*function*

```python
def _scaled_depth(tau: F, omega: F, forward: F)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L36-L38)

## particula.optics.gpu._shortwave.begin_solve

*function*

```python
def begin_solve(status: wp.array2d(dtype=int))
```

Mark every lane pending until all its angular launches have completed.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L318-L322)

## particula.optics.gpu._shortwave.difference_flux

*function*

```python
def difference_flux(total: wp.array2d(dtype=wp.vec3d), refs: wp.array(dtype=int), delta: wp.array2d(dtype=wp.vec3d), status: wp.array2d(dtype=int))
```

Subtract reference broadband flux at the same interface [W m⁻²].

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L369-L381)

## particula.optics.gpu._shortwave.integrate_flux

*function*

```python
def integrate_flux(flux: wp.array3d(dtype=wp.vec3d), weights: wp.array2d(dtype=F), total: wp.array2d(dtype=wp.vec3d), status: wp.array2d(dtype=int))
```

Integrate the three spectral fluxes at each column interface.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L335-L349)

## particula.optics.gpu._shortwave.integrate_radiance

*function*

```python
def integrate_radiance(radiance: wp.array2d(dtype=F), response: wp.array2d(dtype=F), output: wp.array(dtype=F), status: wp.array2d(dtype=int))
```

Apply normalized wavelength weights to each directional spectrum.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L352-L366)

## particula.optics.gpu._shortwave.transport_kernel

*function* · *cached*

```python
def transport_kernel(n)
```

Specialize local dense matrices to the number of angular ordinates.

[View source](https://github.com/uncscode/particula/blob/main/particula/optics/gpu/_shortwave.py#L41-L315)
