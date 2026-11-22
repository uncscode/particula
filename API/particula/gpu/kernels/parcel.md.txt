# particula.gpu.kernels.parcel

*module*

Float64 Warp integration for the warm parcel in dynamics.parcel.

Each time level owns disjoint rows of the trajectory and adjoint arrays.
Kernels read the previous level and write the next. A linearly implicit
Köhler relaxation stabilizes the steep subsaturated equilibrium branch.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L1-L1)

## particula.gpu.kernels.parcel.D_EXPONENT

*attribute* · *module attribute*

```python
D_EXPONENT = wp.constant(F(1.94))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L40-L40)

## particula.gpu.kernels.parcel.D_REF

*attribute* · *module attribute*

```python
D_REF = wp.constant(F(2.11e-05))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L39-L39)

## particula.gpu.kernels.parcel.EPSILON

*attribute* · *module attribute*

```python
EPSILON = wp.constant(F(RD / RV))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L33-L33)

## particula.gpu.kernels.parcel.ES_REF

*attribute* · *module attribute*

```python
ES_REF = wp.constant(F(611.2))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L38-L38)

## particula.gpu.kernels.parcel.F

*attribute* · *module attribute*

```python
F = wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L30-L30)

## particula.gpu.kernels.parcel.GRAVITY

*attribute* · *module attribute*

```python
GRAVITY = wp.constant(F(STANDARD_GRAVITY))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L36-L36)

## particula.gpu.kernels.parcel.KELVIN_T

*attribute* · *module attribute*

```python
KELVIN_T = wp.constant(F(2 * SIGMA / RV * RHO_W))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L31-L31)

## particula.gpu.kernels.parcel.LATENT_OVER_RV

*attribute* · *module attribute*

```python
LATENT_OVER_RV = wp.constant(F(LATENT / RV))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L32-L32)

## particula.gpu.kernels.parcel.R_DRY

*attribute* · *module attribute*

```python
R_DRY = wp.constant(F(RD))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L34-L34)

## particula.gpu.kernels.parcel.R_VAPOR

*attribute* · *module attribute*

```python
R_VAPOR = wp.constant(F(RV))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L35-L35)

## particula.gpu.kernels.parcel.T_REF

*attribute* · *module attribute*

```python
T_REF = wp.constant(F(273.15))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L37-L37)

## particula.gpu.kernels.parcel._advance

*function*

```python
def _advance(previous: FloatVector, liquid: FloatVector, speed: FloatVector, total_water: F, dry_mass: F, dt: F, result: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L132-L161)

## particula.gpu.kernels.parcel._es

*function*

```python
def _es(t: F)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L43-L45)

## particula.gpu.kernels.parcel._grow

*function*

```python
def _grow(previous: FloatVector, radius: FloatVector, dry: FloatVector, kappa: F, dt: F, active: bool, result: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L93-L112)

## particula.gpu.kernels.parcel._grown_radius

*function*

```python
def _grown_radius(r: F, d: F, kappa: F, t: F, p: F, rh: F, dt: F, bound_dry: bool)
```

One linearly implicit water-growth step, shared with forced replay.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L64-L90)

## particula.gpu.kernels.parcel._growth_resistance

*function*

```python
def _growth_resistance(t: F, p: F, es: F)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L58-L61)

## particula.gpu.kernels.parcel._liquid_water

*function*

```python
def _liquid_water(wet: FloatVector, dry: FloatVector, weights: FloatVector, result: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L115-L129)

## particula.gpu.kernels.parcel._observe

*function*

```python
def _observe(state: FloatVector, loss: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L164-L166)

## particula.gpu.kernels.parcel._resistance_terms

*function*

```python
def _resistance_terms(t: F, p: F, es: F)
```

Diffusive and thermal components before multiplying by water density.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L48-L55)

## particula.gpu.kernels.parcel.run_warp_parcel

*function*

```python
def run_warp_parcel(initial, wet, time, updraft, device, gradient, phase_exchange)
```

Advance from equilibrated radii, reading host outputs after the tape.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel.py#L169-L250)
