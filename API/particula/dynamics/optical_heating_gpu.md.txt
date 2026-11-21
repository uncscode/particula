# particula.dynamics.optical_heating_gpu

*module*

Resident optical heating using staged particle and box inventories.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L1-L1)

## particula.dynamics.optical_heating_gpu._FOUR

*attribute* · *module attribute*

```python
_FOUR = wp.constant(wp.float64(4.0))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L35-L35)

## particula.dynamics.optical_heating_gpu._MAX

*attribute* · *module attribute*

```python
_MAX = wp.constant(wp.float64(_MAX_K))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L30-L30)

## particula.dynamics.optical_heating_gpu._MIN

*attribute* · *module attribute*

```python
_MIN = wp.constant(wp.float64(_MIN_K))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L29-L29)

## particula.dynamics.optical_heating_gpu._PI

*attribute* · *module attribute*

```python
_PI = wp.constant(wp.float64(pi))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L28-L28)

## particula.dynamics.optical_heating_gpu._R

*attribute* · *module attribute*

```python
_R = wp.constant(wp.float64(GAS_CONSTANT))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L32-L32)

## particula.dynamics.optical_heating_gpu._SIGMA

*attribute* · *module attribute*

```python
_SIGMA = wp.constant(wp.float64(Stefan_Boltzmann))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L31-L31)

## particula.dynamics.optical_heating_gpu._TWO

*attribute* · *module attribute*

```python
_TWO = wp.constant(wp.float64(2.0))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L34-L34)

## particula.dynamics.optical_heating_gpu._ZERO

*attribute* · *module attribute*

```python
_ZERO = wp.constant(wp.float64(0.0))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L33-L33)

## particula.dynamics.optical_heating_gpu.OpticalHeatingAdapter

*class* · *dataclass*

```python
class OpticalHeatingAdapter
```

Bind optical heating to Warp with bounded coefficient-series scratch.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L343-L514)

### particula.dynamics.optical_heating_gpu.OpticalHeatingAdapter.coefficient_capacity

*attribute* · *class attribute* · *instance attribute*

```python
coefficient_capacity: int = 512
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L347-L347)

### particula.dynamics.optical_heating_gpu.OpticalHeatingAdapter.particle_tile

*attribute* · *class attribute* · *instance attribute*

```python
particle_tile: int = 128
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L348-L348)

### particula.dynamics.optical_heating_gpu.OpticalHeatingAdapter.__init__

*method*

```python
def __init__(coefficient_capacity: int = 512, particle_tile: int = 128) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L0-L0)

### particula.dynamics.optical_heating_gpu.OpticalHeatingAdapter.plan

*method*

```python
def plan(process, aerosol)
```

Validate process identity, native fields and workspace limits.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L350-L382)

### particula.dynamics.optical_heating_gpu.OpticalHeatingAdapter.prepare

*method*

```python
def prepare(plan, context)
```

Retain an optical workspace, then stage and commit on its stream.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L384-L514)

## particula.dynamics.optical_heating_gpu._Balance

*class*

```python
class _Balance
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L59-L71)

### particula.dynamics.optical_heating_gpu._Balance.conduction

*attribute* · *instance attribute*

```python
conduction: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L67-L67)

### particula.dynamics.optical_heating_gpu._Balance.emission

*attribute* · *instance attribute*

```python
emission: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L68-L68)

### particula.dynamics.optical_heating_gpu._Balance.gas_temperature

*attribute* · *instance attribute*

```python
gas_temperature: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L71-L71)

### particula.dynamics.optical_heating_gpu._Balance.gas_vapor

*attribute* · *instance attribute*

```python
gas_vapor: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L63-L63)

### particula.dynamics.optical_heating_gpu._Balance.heat_capacity

*attribute* · *instance attribute*

```python
heat_capacity: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L66-L66)

### particula.dynamics.optical_heating_gpu._Balance.latent_heat

*attribute* · *instance attribute*

```python
latent_heat: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L65-L65)

### particula.dynamics.optical_heating_gpu._Balance.old_temperature

*attribute* · *instance attribute*

```python
old_temperature: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L70-L70)

### particula.dynamics.optical_heating_gpu._Balance.radiative_background

*attribute* · *instance attribute*

```python
radiative_background: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L69-L69)

### particula.dynamics.optical_heating_gpu._Balance.saturation_to_density

*attribute* · *instance attribute*

```python
saturation_to_density: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L62-L62)

### particula.dynamics.optical_heating_gpu._Balance.transfer

*attribute* · *instance attribute*

```python
transfer: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L64-L64)

### particula.dynamics.optical_heating_gpu._Balance.water

*attribute* · *instance attribute*

```python
water: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L61-L61)

## particula.dynamics.optical_heating_gpu._BoxEnergy

*class*

```python
class _BoxEnergy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L50-L56)

### particula.dynamics.optical_heating_gpu._BoxEnergy.absorbed

*attribute* · *instance attribute*

```python
absorbed: wp.array(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L52-L52)

### particula.dynamics.optical_heating_gpu._BoxEnergy.conduction

*attribute* · *instance attribute*

```python
conduction: wp.array(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L53-L53)

### particula.dynamics.optical_heating_gpu._BoxEnergy.latent

*attribute* · *instance attribute*

```python
latent: wp.array(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L55-L55)

### particula.dynamics.optical_heating_gpu._BoxEnergy.radiation

*attribute* · *instance attribute*

```python
radiation: wp.array(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L54-L54)

### particula.dynamics.optical_heating_gpu._BoxEnergy.sensible

*attribute* · *instance attribute*

```python
sensible: wp.array(dtype=wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L56-L56)

## particula.dynamics.optical_heating_gpu._Settings

*class*

```python
class _Settings
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L38-L47)

### particula.dynamics.optical_heating_gpu._Settings.conductivity

*attribute* · *instance attribute*

```python
conductivity: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L41-L41)

### particula.dynamics.optical_heating_gpu._Settings.cp_solute

*attribute* · *instance attribute*

```python
cp_solute: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L44-L44)

### particula.dynamics.optical_heating_gpu._Settings.cp_water

*attribute* · *instance attribute*

```python
cp_water: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L43-L43)

### particula.dynamics.optical_heating_gpu._Settings.diffusivity

*attribute* · *instance attribute*

```python
diffusivity: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L42-L42)

### particula.dynamics.optical_heating_gpu._Settings.emissivity

*attribute* · *instance attribute*

```python
emissivity: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L46-L46)

### particula.dynamics.optical_heating_gpu._Settings.kappa

*attribute* · *instance attribute*

```python
kappa: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L40-L40)

### particula.dynamics.optical_heating_gpu._Settings.latent

*attribute* · *instance attribute*

```python
latent: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L45-L45)

### particula.dynamics.optical_heating_gpu._Settings.radiation_temperature

*attribute* · *instance attribute*

```python
radiation_temperature: wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L47-L47)

## particula.dynamics.optical_heating_gpu._absorbed_power

*function*

```python
def _absorbed_power(box: int, slot: int, n_slots: int, cross_area: wp.float64, irradiance: wp.array(dtype=wp.float64), efficiencies: wp.array2d(dtype=wp.float64), optical_status: wp.array(dtype=wp.int32)) -> wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L107-L124)

## particula.dynamics.optical_heating_gpu._fourth

*function*

```python
def _fourth(value: wp.float64) -> wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L74-L77)

## particula.dynamics.optical_heating_gpu._optical_config

*function*

```python
def _optical_config(metadata)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L330-L340)

## particula.dynamics.optical_heating_gpu._publish

*function*

```python
def _publish(particles: WarpParticleData, gas: wp.array2d(dtype=wp.float64), temperature: wp.array2d(dtype=wp.float64), power: wp.array2d(dtype=wp.float64), ledger: _BoxEnergy, particle_water: int, gas_water: int, proposed: wp.array3d(dtype=wp.float64), next_box: wp.array2d(dtype=wp.float64), failure: wp.array(dtype=wp.int32))
```

Commit both phases and the energy ledger after a successful batch.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L299-L327)

## particula.dynamics.optical_heating_gpu._stage_box

*function*

```python
def _stage_box(particles: WarpParticleData, gas: wp.array2d(dtype=wp.float64), ledger: _BoxEnergy, weights: wp.array2d(dtype=wp.float64), waves: int, particle_water: int, gas_water: int, proposed: wp.array3d(dtype=wp.float64), next_box: wp.array2d(dtype=wp.float64), failure: wp.array(dtype=wp.int32))
```

Form extensive changes in a fixed order before publishing any box.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L241-L296)

## particula.dynamics.optical_heating_gpu._stage_particle

*function*

```python
def _stage_particle(particles: WarpParticleData, gas: wp.array2d(dtype=wp.float64), molar_mass: wp.array(dtype=wp.float64), ambient: wp.array(dtype=wp.float64), temperature: wp.array2d(dtype=wp.float64), settings: _Settings, particle_water: int, gas_water: int, dt: wp.float64, irradiance: wp.array(dtype=wp.float64), radii: wp.array2d(dtype=wp.float64), efficiencies: wp.array2d(dtype=wp.float64), optical_status: wp.array(dtype=wp.int32), proposed: wp.array3d(dtype=wp.float64), failure: wp.array(dtype=wp.int32))
```

Propose each active particle's temperature, transfer and energies.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L146-L238)

## particula.dynamics.optical_heating_gpu._sum4

*function*

```python
def _sum4(x: wp.vec4d) -> wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L102-L104)

## particula.dynamics.optical_heating_gpu._temperature_root

*function*

```python
def _temperature_root(balance: _Balance, absorbed: wp.float64) -> wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L127-L143)

## particula.dynamics.optical_heating_gpu._terms

*function*

```python
def _terms(temperature: wp.float64, balance: _Balance)
```

Evaluate step energies and the evaporation proposal at temperature.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating_gpu.py#L80-L99)
