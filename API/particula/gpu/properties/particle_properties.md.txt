# particula.gpu.properties.particle_properties

*module*

Device-native fp64 particle transport and thermodynamic properties.

The ``@wp.func`` helpers provide scalar SI-unit property calculations for Warp
kernels. Alongside GPU counterparts of CPU particle properties, this module
owns the reusable neutral transport primitives and dimensionless wall-loss
geometry factors used by GPU dynamics.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L1-L1)

## particula.gpu.properties.particle_properties._CUNNINGHAM_CONSTANT

*attribute* · *module attribute*

```python
_CUNNINGHAM_CONSTANT = wp.constant(wp.float64(1.257))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L20-L20)

## particula.gpu.properties.particle_properties._CUNNINGHAM_EXPONENTIAL_RATE

*attribute* · *module attribute*

```python
_CUNNINGHAM_EXPONENTIAL_RATE = wp.constant(wp.float64(1.1))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L22-L22)

## particula.gpu.properties.particle_properties._CUNNINGHAM_EXPONENTIAL_SCALE

*attribute* · *module attribute*

```python
_CUNNINGHAM_EXPONENTIAL_SCALE = wp.constant(wp.float64(0.4))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L21-L21)

## particula.gpu.properties.particle_properties._DEBYE_NODE_01

*attribute* · *module attribute*

```python
_DEBYE_NODE_01 = wp.constant(wp.float64(0.0483076656877383))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L28-L28)

## particula.gpu.properties.particle_properties._DEBYE_NODE_02

*attribute* · *module attribute*

```python
_DEBYE_NODE_02 = wp.constant(wp.float64(0.1444719615827965))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L30-L30)

## particula.gpu.properties.particle_properties._DEBYE_NODE_03

*attribute* · *module attribute*

```python
_DEBYE_NODE_03 = wp.constant(wp.float64(0.2392873622521371))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L32-L32)

## particula.gpu.properties.particle_properties._DEBYE_NODE_04

*attribute* · *module attribute*

```python
_DEBYE_NODE_04 = wp.constant(wp.float64(0.3318686022821277))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L34-L34)

## particula.gpu.properties.particle_properties._DEBYE_NODE_05

*attribute* · *module attribute*

```python
_DEBYE_NODE_05 = wp.constant(wp.float64(0.4213512761306353))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L36-L36)

## particula.gpu.properties.particle_properties._DEBYE_NODE_06

*attribute* · *module attribute*

```python
_DEBYE_NODE_06 = wp.constant(wp.float64(0.5068999089322294))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L38-L38)

## particula.gpu.properties.particle_properties._DEBYE_NODE_07

*attribute* · *module attribute*

```python
_DEBYE_NODE_07 = wp.constant(wp.float64(0.5877157572407623))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L40-L40)

## particula.gpu.properties.particle_properties._DEBYE_NODE_08

*attribute* · *module attribute*

```python
_DEBYE_NODE_08 = wp.constant(wp.float64(0.6630442669302152))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L42-L42)

## particula.gpu.properties.particle_properties._DEBYE_NODE_09

*attribute* · *module attribute*

```python
_DEBYE_NODE_09 = wp.constant(wp.float64(0.7321821187402897))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L44-L44)

## particula.gpu.properties.particle_properties._DEBYE_NODE_10

*attribute* · *module attribute*

```python
_DEBYE_NODE_10 = wp.constant(wp.float64(0.7944837959679424))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L46-L46)

## particula.gpu.properties.particle_properties._DEBYE_NODE_11

*attribute* · *module attribute*

```python
_DEBYE_NODE_11 = wp.constant(wp.float64(0.84936761373257))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L48-L48)

## particula.gpu.properties.particle_properties._DEBYE_NODE_12

*attribute* · *module attribute*

```python
_DEBYE_NODE_12 = wp.constant(wp.float64(0.8963211557660521))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L50-L50)

## particula.gpu.properties.particle_properties._DEBYE_NODE_13

*attribute* · *module attribute*

```python
_DEBYE_NODE_13 = wp.constant(wp.float64(0.9349060759377397))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L52-L52)

## particula.gpu.properties.particle_properties._DEBYE_NODE_14

*attribute* · *module attribute*

```python
_DEBYE_NODE_14 = wp.constant(wp.float64(0.9647622555875064))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L54-L54)

## particula.gpu.properties.particle_properties._DEBYE_NODE_15

*attribute* · *module attribute*

```python
_DEBYE_NODE_15 = wp.constant(wp.float64(0.9856115115452684))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L56-L56)

## particula.gpu.properties.particle_properties._DEBYE_NODE_16

*attribute* · *module attribute*

```python
_DEBYE_NODE_16 = wp.constant(wp.float64(0.9972638618494816))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L58-L58)

## particula.gpu.properties.particle_properties._DEBYE_WEIGHT_01

*attribute* · *module attribute*

```python
_DEBYE_WEIGHT_01 = wp.constant(wp.float64(0.0965400885147278))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L29-L29)

## particula.gpu.properties.particle_properties._DEBYE_WEIGHT_02

*attribute* · *module attribute*

```python
_DEBYE_WEIGHT_02 = wp.constant(wp.float64(0.0956387200792749))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L31-L31)

## particula.gpu.properties.particle_properties._DEBYE_WEIGHT_03

*attribute* · *module attribute*

```python
_DEBYE_WEIGHT_03 = wp.constant(wp.float64(0.0938443990808046))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L33-L33)

## particula.gpu.properties.particle_properties._DEBYE_WEIGHT_04

*attribute* · *module attribute*

```python
_DEBYE_WEIGHT_04 = wp.constant(wp.float64(0.0911738786957639))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L35-L35)

## particula.gpu.properties.particle_properties._DEBYE_WEIGHT_05

*attribute* · *module attribute*

```python
_DEBYE_WEIGHT_05 = wp.constant(wp.float64(0.0876520930044038))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L37-L37)

## particula.gpu.properties.particle_properties._DEBYE_WEIGHT_06

*attribute* · *module attribute*

```python
_DEBYE_WEIGHT_06 = wp.constant(wp.float64(0.0833119242269468))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L39-L39)

## particula.gpu.properties.particle_properties._DEBYE_WEIGHT_07

*attribute* · *module attribute*

```python
_DEBYE_WEIGHT_07 = wp.constant(wp.float64(0.0781938957870703))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L41-L41)

## particula.gpu.properties.particle_properties._DEBYE_WEIGHT_08

*attribute* · *module attribute*

```python
_DEBYE_WEIGHT_08 = wp.constant(wp.float64(0.0723457941088485))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L43-L43)

## particula.gpu.properties.particle_properties._DEBYE_WEIGHT_09

*attribute* · *module attribute*

```python
_DEBYE_WEIGHT_09 = wp.constant(wp.float64(0.0658222227763618))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L45-L45)

## particula.gpu.properties.particle_properties._DEBYE_WEIGHT_10

*attribute* · *module attribute*

```python
_DEBYE_WEIGHT_10 = wp.constant(wp.float64(0.0586840934785355))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L47-L47)

## particula.gpu.properties.particle_properties._DEBYE_WEIGHT_11

*attribute* · *module attribute*

```python
_DEBYE_WEIGHT_11 = wp.constant(wp.float64(0.0509980592623762))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L49-L49)

## particula.gpu.properties.particle_properties._DEBYE_WEIGHT_12

*attribute* · *module attribute*

```python
_DEBYE_WEIGHT_12 = wp.constant(wp.float64(0.0428358980222267))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L51-L51)

## particula.gpu.properties.particle_properties._DEBYE_WEIGHT_13

*attribute* · *module attribute*

```python
_DEBYE_WEIGHT_13 = wp.constant(wp.float64(0.0342738629130214))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L53-L53)

## particula.gpu.properties.particle_properties._DEBYE_WEIGHT_14

*attribute* · *module attribute*

```python
_DEBYE_WEIGHT_14 = wp.constant(wp.float64(0.0253920653092621))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L55-L55)

## particula.gpu.properties.particle_properties._DEBYE_WEIGHT_15

*attribute* · *module attribute*

```python
_DEBYE_WEIGHT_15 = wp.constant(wp.float64(0.0162743947309057))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L57-L57)

## particula.gpu.properties.particle_properties._DEBYE_WEIGHT_16

*attribute* · *module attribute*

```python
_DEBYE_WEIGHT_16 = wp.constant(wp.float64(0.0070186100094701))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L59-L59)

## particula.gpu.properties.particle_properties._FUCHS_SUTUGIN_DENOMINATOR

*attribute* · *module attribute*

```python
_FUCHS_SUTUGIN_DENOMINATOR = wp.constant(wp.float64(0.283))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L27-L27)

## particula.gpu.properties.particle_properties._FUCHS_SUTUGIN_NUMERATOR

*attribute* · *module attribute*

```python
_FUCHS_SUTUGIN_NUMERATOR = wp.constant(wp.float64(0.75))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L26-L26)

## particula.gpu.properties.particle_properties._MIN_NORMAL_FLOAT64

*attribute* · *module attribute*

```python
_MIN_NORMAL_FLOAT64 = wp.constant(wp.float64(2.2250738585072014e-308))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L24-L24)

## particula.gpu.properties.particle_properties._PI

*attribute* · *module attribute*

```python
_PI = wp.constant(wp.float64(3.141592653589793))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L19-L19)

## particula.gpu.properties.particle_properties._PI_SQUARED_OVER_SIX

*attribute* · *module attribute*

```python
_PI_SQUARED_OVER_SIX = wp.constant(wp.float64(1.6449340668482264))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L25-L25)

## particula.gpu.properties.particle_properties._STANDARD_GRAVITY

*attribute* · *module attribute*

```python
_STANDARD_GRAVITY = wp.constant(wp.float64(STANDARD_GRAVITY))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L23-L23)

## particula.gpu.properties.particle_properties._debye_gauss_pair_wp

*function*

```python
def _debye_gauss_pair_wp(half_x: wp.float64, midpoint: wp.float64, node: wp.float64, weight: wp.float64) -> wp.float64
```

Return one symmetric Gauss--Legendre pair contribution.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L351-L362)

## particula.gpu.properties.particle_properties._debye_integrand_wp

*function*

```python
def _debye_integrand_wp(value: wp.float64) -> wp.float64
```

Evaluate the removable Debye integrand ``t / (exp(t) - 1)``.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L343-L348)

## particula.gpu.properties.particle_properties.aerodynamic_mobility_wp

*function*

```python
def aerodynamic_mobility_wp(particle_radius: wp.float64, slip_correction_factor: wp.float64, dynamic_viscosity: wp.float64) -> wp.float64
```

Calculate aerodynamic mobility of a particle.

Port of
``particula.particles.properties.aerodynamic_mobility_module.get_aerodynamic_mobility``.

**Parameters**

- `particle_radius` (`wp.float64`) — Particle radius [m].
- `slip_correction_factor` (`wp.float64`) — Slip correction factor (dimensionless).
- `dynamic_viscosity` (`wp.float64`) — Dynamic viscosity [Pa·s].

**Returns**

- (`wp.float64`) — Aerodynamic mobility [m²/s].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L559-L581)

## particula.gpu.properties.particle_properties.cunningham_slip_correction_wp

*function*

```python
def cunningham_slip_correction_wp(knudsen_number: wp.float64) -> wp.float64
```

Calculate the Cunningham slip correction from Knudsen number.

Evaluates ``C_c = 1 + Kn * (1.257 + 0.4 * exp(-1.1 / Kn))``. The
continuum limit at ``Kn = 0`` is exactly ``1.0``; exponential underflow
for small positive ``Kn`` is an intended path. Negative and non-finite
inputs return exact ``0.0`` before division.

**Parameters**

- `knudsen_number` (`wp.float64`) — Particle Knudsen number ``Kn`` (dimensionless).

**Returns**

- (`wp.float64`) — Dimensionless slip correction, or the documented exact limit or
- (`wp.float64`) — invalid-input sentinel.

> **References**
>
> Seinfeld, J. H., & Pandis, S. N. (2016). *Atmospheric Chemistry and
> Physics* (3rd ed.). John Wiley & Sons.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L81-L113)

## particula.gpu.properties.particle_properties.debye_1_wp

*function*

```python
def debye_1_wp(x: wp.float64) -> wp.float64
```

Calculate the dimensionless first Debye wall-loss geometry factor.

Evaluates ``D_1(x) = integral_0^x[t / (exp(t) - 1)] dt / x``. Its zero
limit is exactly ``1.0``; negative and non-finite inputs return exact
``0.0``. The implementation uses a Bernoulli series for ``x <= 1``,
32-node Gauss--Legendre quadrature for ``1 < x < 20``, and the
finite-tail-corrected ``pi² / (6 x)`` asymptote at larger ``x``.

**Parameters**

- `x` (`wp.float64`) — Dimensionless spherical wall-loss geometry argument.

**Returns**

- (`wp.float64`) — Dimensionless ``D_1(x)`` value, or the documented exact limit or
- (`wp.float64`) — invalid-input sentinel.

> **References**
>
> Crump, J. G., Flagan, R. C., & Seinfeld, J. H. (1982). Particle wall
> loss rates in vessels. *Aerosol Science and Technology*, 2(3), 303-309.
> https://doi.org/10.1080/02786828308958636

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L365-L519)

## particula.gpu.properties.particle_properties.diffusion_coefficient_wp

*function*

```python
def diffusion_coefficient_wp(temperature: wp.float64, aerodynamic_mobility: wp.float64, boltzmann_constant: wp.float64) -> wp.float64
```

Calculate Stokes-Einstein diffusion from mobility.

Evaluates ``D = k_B T B``. This low-level device primitive intentionally
performs no domain checks; callers must provide physically valid SI inputs.

**Parameters**

- `temperature` (`wp.float64`) — Gas temperature ``T`` [K].
- `aerodynamic_mobility` (`wp.float64`) — Particle mobility ``B`` [s/kg].
- `boltzmann_constant` (`wp.float64`) — Boltzmann constant ``k_B`` [J/K].

**Returns**

- (`wp.float64`) — Particle diffusion coefficient ``D`` [m²/s].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L138-L157)

## particula.gpu.properties.particle_properties.effective_density_wp

*function*

```python
def effective_density_wp(total_mass: wp.float64, total_volume: wp.float64) -> wp.float64
```

Calculate a safe spherical-equivalent mixture density.

Evaluates ``rho_eff = m / V`` from total mass and volume. Non-finite,
non-positive, subnormal, or overflowed inputs and results return exact
``0.0``.

**Parameters**

- `total_mass` (`wp.float64`) — Total particle mass [kg].
- `total_volume` (`wp.float64`) — Total particle volume [m³].

**Returns**

- (`wp.float64`) — Effective mixture density [kg/m³], or exact ``0.0`` for an invalid or
- (`wp.float64`) — non-normal calculation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L160-L194)

## particula.gpu.properties.particle_properties.friction_factor_wp

*function*

```python
def friction_factor_wp(particle_radius: wp.float64, dynamic_viscosity: wp.float64, slip_correction: wp.float64) -> wp.float64
```

Calculate the friction factor for a particle.

Port of
``particula.particles.properties.friction_factor_module.get_friction_factor``.

**Parameters**

- `particle_radius` (`wp.float64`) — Particle radius [m].
- `dynamic_viscosity` (`wp.float64`) — Dynamic viscosity [Pa·s].
- `slip_correction` (`wp.float64`) — Slip correction factor (dimensionless).

**Returns**

- (`wp.float64`) — Friction factor [N·s/m].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L610-L632)

## particula.gpu.properties.particle_properties.kelvin_radius_wp

*function*

```python
def kelvin_radius_wp(effective_surface_tension: wp.float64, effective_density: wp.float64, molar_mass: wp.float64, temperature: wp.float64, gas_constant: wp.float64) -> wp.float64
```

Calculate the Kelvin radius.

Port of
``particula.particles.properties.kelvin_effect_module.get_kelvin_radius``.

**Parameters**

- `effective_surface_tension` (`wp.float64`) — Effective surface tension [N/m].
- `effective_density` (`wp.float64`) — Effective density [kg/m³].
- `molar_mass` (`wp.float64`) — Molar mass [kg/mol].
- `temperature` (`wp.float64`) — Temperature [K].
- `gas_constant` (`wp.float64`) — Gas constant [J/(mol·K)].

**Returns**

- (`wp.float64`) — Kelvin radius [m].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L666-L691)

## particula.gpu.properties.particle_properties.kelvin_term_wp

*function*

```python
def kelvin_term_wp(particle_radius: wp.float64, kelvin_radius_value: wp.float64) -> wp.float64
```

Calculate the Kelvin term with safe clamping.

Port of
``particula.particles.properties.kelvin_effect_module.get_kelvin_term``.

**Parameters**

- `particle_radius` (`wp.float64`) — Particle radius [m].
- `kelvin_radius_value` (`wp.float64`) — Kelvin radius [m].

**Returns**

- (`wp.float64`) — Kelvin term (dimensionless).

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L694-L716)

## particula.gpu.properties.particle_properties.knudsen_number_wp

*function*

```python
def knudsen_number_wp(mean_free_path: wp.float64, particle_radius: wp.float64) -> wp.float64
```

Calculate the Knudsen number from mean free path and radius.

Port of ``particula.particles.properties.knudsen_number_module``.

**Parameters**

- `mean_free_path` (`wp.float64`) — Mean free path of the gas molecules [m].
- `particle_radius` (`wp.float64`) — Particle radius [m].

**Returns**

- (`wp.float64`) — Knudsen number (dimensionless).

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L62-L78)

## particula.gpu.properties.particle_properties.mean_thermal_speed_wp

*function*

```python
def mean_thermal_speed_wp(particle_mass: wp.float64, temperature: wp.float64, boltzmann_constant: wp.float64) -> wp.float64
```

Calculate the mean thermal speed of a particle.

Port of
``particula.particles.properties.mean_thermal_speed_module.get_mean_thermal_speed``.

**Parameters**

- `particle_mass` (`wp.float64`) — Particle mass [kg].
- `temperature` (`wp.float64`) — Gas temperature [K].
- `boltzmann_constant` (`wp.float64`) — Boltzmann constant [J/K].

**Returns**

- (`wp.float64`) — Mean thermal speed [m/s].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L584-L607)

## particula.gpu.properties.particle_properties.partial_pressure_delta_wp

*function*

```python
def partial_pressure_delta_wp(partial_pressure_gas: wp.float64, partial_pressure_particle: wp.float64, kelvin_term: wp.float64) -> wp.float64
```

Calculate the partial pressure delta.

Port of
``particula.particles.properties.partial_pressure_module.get_partial_pressure_delta``.

**Parameters**

- `partial_pressure_gas` (`wp.float64`) — Gas-phase partial pressure [Pa].
- `partial_pressure_particle` (`wp.float64`) — Particle-phase partial pressure [Pa].
- `kelvin_term` (`wp.float64`) — Kelvin term (dimensionless).

**Returns**

- (`wp.float64`) — Partial pressure delta [Pa].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L719-L738)

## particula.gpu.properties.particle_properties.particle_radius_from_volume_wp

*function*

```python
def particle_radius_from_volume_wp(total_volume: wp.float64) -> wp.float64
```

Calculate a spherical particle radius from total volume.

Evaluates ``r = (3 V / (4 pi))**(1/3)`` for volume ``V``. Non-finite and
non-positive volumes return exact ``0.0`` before the cube root.

**Parameters**

- `total_volume` (`wp.float64`) — Total spherical-equivalent particle volume ``V`` [m³].

**Returns**

- (`wp.float64`) — Spherical-equivalent particle radius [m], or exact ``0.0`` for an
- (`wp.float64`) — invalid volume.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L116-L135)

## particula.gpu.properties.particle_properties.settling_velocity_stokes_from_transport_wp

*function*

```python
def settling_velocity_stokes_from_transport_wp(particle_radius: wp.float64, effective_density: wp.float64, dynamic_viscosity: wp.float64, mean_free_path: wp.float64) -> wp.float64
```

Calculate safe slip-corrected Stokes settling velocity.

Evaluates ``v_s = 2 r² rho_eff C_c g / (9 mu)`` from precomputed gas
transport properties and standard gravity. Invalid, non-positive,
subnormal, or non-finite intermediates and results return exact ``0.0``.

**Parameters**

- `particle_radius` (`wp.float64`) — Particle radius [m].
- `effective_density` (`wp.float64`) — Effective particle density [kg/m³].
- `dynamic_viscosity` (`wp.float64`) — Dynamic gas viscosity [Pa s].
- `mean_free_path` (`wp.float64`) — Molecular mean free path [m].

**Returns**

- (`wp.float64`) — Downward settling velocity [m/s], or exact ``0.0`` for an invalid or
- (`wp.float64`) — non-normal calculation.

> **References**
>
> Seinfeld, J. H., & Pandis, S. N. (2016). *Atmospheric Chemistry and
> Physics* (3rd ed.). John Wiley & Sons.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L197-L261)

## particula.gpu.properties.particle_properties.settling_velocity_stokes_wp

*function*

```python
def settling_velocity_stokes_wp(particle_radius: wp.float64, effective_density: wp.float64, temperature: wp.float64, pressure: wp.float64, gas_constant: wp.float64, molecular_weight_air: wp.float64, ref_viscosity: wp.float64, ref_temperature: wp.float64, sutherland_constant: wp.float64) -> wp.float64
```

Calculate safe slip-corrected Stokes settling velocity from gas state.

Derives viscosity, molecular mean free path, Knudsen number, and slip
correction from the supplied SI gas state before evaluating
``v_s = 2 r² rho_eff C_c g / (9 mu)``. Invalid inputs or derived transport
return exact ``0.0``.

**Parameters**

- `particle_radius` (`wp.float64`) — Particle radius [m].
- `effective_density` (`wp.float64`) — Effective particle density [kg/m³].
- `temperature` (`wp.float64`) — Gas temperature [K].
- `pressure` (`wp.float64`) — Gas pressure [Pa].
- `gas_constant` (`wp.float64`) — Universal gas constant [J/(mol K)].
- `molecular_weight_air` (`wp.float64`) — Air molar mass [kg/mol].
- `ref_viscosity` (`wp.float64`) — Reference dynamic viscosity [Pa s].
- `ref_temperature` (`wp.float64`) — Reference temperature [K].
- `sutherland_constant` (`wp.float64`) — Sutherland temperature constant [K].

**Returns**

- (`wp.float64`) — Downward settling velocity [m/s], or exact ``0.0`` for invalid input
- (`wp.float64`) — or derived transport.

> **References**
>
> Seinfeld, J. H., & Pandis, S. N. (2016). *Atmospheric Chemistry and
> Physics* (3rd ed.). John Wiley & Sons.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L264-L340)

## particula.gpu.properties.particle_properties.vapor_transition_correction_wp

*function*

```python
def vapor_transition_correction_wp(knudsen_number: wp.float64, mass_accommodation: wp.float64) -> wp.float64
```

Calculate the vapor transition correction factor.

Port of
``particula.particles.properties.vapor_correction_module.get_vapor_transition_correction``.

**Parameters**

- `knudsen_number` (`wp.float64`) — Knudsen number (dimensionless).
- `mass_accommodation` (`wp.float64`) — Mass accommodation coefficient (dimensionless).

**Returns**

- (`wp.float64`) — Vapor transition correction factor (dimensionless).

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L635-L663)

## particula.gpu.properties.particle_properties.x_coth_x_wp

*function*

```python
def x_coth_x_wp(x: wp.float64) -> wp.float64
```

Calculate the rectangular wall-loss factor ``x / tanh(x)``.

The zero limit is exactly ``1.0``; negative and non-finite inputs return
exact ``0.0``. A series through ``x**6`` for ``0 < x <= 1e-3`` avoids
cancellation. The factor supports the rearranged rectangular term
``(4 * sqrt(K * D) / pi) * x_coth_x_wp(x)``.

**Parameters**

- `x` (`wp.float64`) — Dimensionless rectangular wall-loss geometry argument.

**Returns**

- (`wp.float64`) — Dimensionless ``x / tanh(x)`` value, or the documented exact limit or
- (`wp.float64`) — invalid-input sentinel.

> **References**
>
> Crump, J. G., Flagan, R. C., & Seinfeld, J. H. (1982). Particle wall
> loss rates in vessels. *Aerosol Science and Technology*, 2(3), 303-309.
> https://doi.org/10.1080/02786828308958636

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L522-L556)
