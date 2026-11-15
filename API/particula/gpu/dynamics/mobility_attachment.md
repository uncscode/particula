# particula.gpu.dynamics.mobility_attachment

*module*

Device ion transport approximation; see dynamics.ions.mobility_attachment.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/mobility_attachment.py#L1-L1)

## particula.gpu.dynamics.mobility_attachment._CUTOFF

*attribute* · *module attribute*

```python
_CUTOFF = wp.constant(wp.float64(1e-80))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/mobility_attachment.py#L45-L45)

## particula.gpu.dynamics.mobility_attachment._E

*attribute* · *module attribute*

```python
_E = wp.constant(wp.float64(ELEMENTARY_CHARGE_VALUE))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/mobility_attachment.py#L31-L31)

## particula.gpu.dynamics.mobility_attachment._EPS

*attribute* · *module attribute*

```python
_EPS = wp.constant(wp.float64(ELECTRIC_PERMITTIVITY))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/mobility_attachment.py#L32-L32)

## particula.gpu.dynamics.mobility_attachment._FIT_A

*attribute* · *module attribute*

```python
_FIT_A = wp.constant(wp.float64(25.836))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/mobility_attachment.py#L41-L41)

## particula.gpu.dynamics.mobility_attachment._FIT_B

*attribute* · *module attribute*

```python
_FIT_B = wp.constant(wp.float64(11.211))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/mobility_attachment.py#L42-L42)

## particula.gpu.dynamics.mobility_attachment._FIT_C

*attribute* · *module attribute*

```python
_FIT_C = wp.constant(wp.float64(3.502))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/mobility_attachment.py#L43-L43)

## particula.gpu.dynamics.mobility_attachment._FIT_D

*attribute* · *module attribute*

```python
_FIT_D = wp.constant(wp.float64(7.211))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/mobility_attachment.py#L44-L44)

## particula.gpu.dynamics.mobility_attachment._ION_TREF

*attribute* · *module attribute*

```python
_ION_TREF = wp.constant(wp.float64(298.15))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/mobility_attachment.py#L40-L40)

## particula.gpu.dynamics.mobility_attachment._K

*attribute* · *module attribute*

```python
_K = wp.constant(wp.float64(BOLTZMANN_CONSTANT))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/mobility_attachment.py#L30-L30)

## particula.gpu.dynamics.mobility_attachment._MAIR

*attribute* · *module attribute*

```python
_MAIR = wp.constant(wp.float64(MOLECULAR_WEIGHT_AIR))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/mobility_attachment.py#L34-L34)

## particula.gpu.dynamics.mobility_attachment._MU

*attribute* · *module attribute*

```python
_MU = wp.constant(wp.float64(REF_VISCOSITY_AIR_STP))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/mobility_attachment.py#L35-L35)

## particula.gpu.dynamics.mobility_attachment._PI

*attribute* · *module attribute*

```python
_PI = wp.constant(wp.float64(3.141592653589793))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/mobility_attachment.py#L38-L38)

## particula.gpu.dynamics.mobility_attachment._R

*attribute* · *module attribute*

```python
_R = wp.constant(wp.float64(GAS_CONSTANT))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/mobility_attachment.py#L33-L33)

## particula.gpu.dynamics.mobility_attachment._S

*attribute* · *module attribute*

```python
_S = wp.constant(wp.float64(SUTHERLAND_CONSTANT))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/mobility_attachment.py#L37-L37)

## particula.gpu.dynamics.mobility_attachment._TREF

*attribute* · *module attribute*

```python
_TREF = wp.constant(wp.float64(REF_TEMPERATURE_STP))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/mobility_attachment.py#L36-L36)

## particula.gpu.dynamics.mobility_attachment.mobility_attachment_wp

*function*

```python
def mobility_attachment_wp(radius: wp.float64, mass: wp.float64, charge: wp.float64, temperature: wp.float64, pressure: wp.float64, ion_mass: wp.float64, ion_mobility: wp.float64, ion_sign: wp.float64, ion_radius: wp.float64) -> wp.float64
```

Return SI attachment rate, or -1 for unsupported numeric inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/mobility_attachment.py#L48-L115)
