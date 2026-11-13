# particula.gpu.properties

*module*

Public Warp fp64 gas and particle property helpers.

This package exports device-only scalar property functions for use inside Warp
kernels. Neutral transport and wall-loss geometry helpers are canonically owned
by :mod:`particula.gpu.properties.particle_properties`.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/__init__.py#L1-L1)

## particula.gpu.properties.aerodynamic_mobility_wp

*function*

```python
def aerodynamic_mobility_wp(particle_radius: wp.float64, slip_correction_factor: wp.float64, dynamic_viscosity: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.properties.particle_properties`

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

## particula.gpu.properties.cunningham_slip_correction_wp

*function*

```python
def cunningham_slip_correction_wp(knudsen_number: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.properties.particle_properties`

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

## particula.gpu.properties.debye_1_wp

*function*

```python
def debye_1_wp(x: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.properties.particle_properties`

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

## particula.gpu.properties.diffusion_coefficient_wp

*function*

```python
def diffusion_coefficient_wp(temperature: wp.float64, aerodynamic_mobility: wp.float64, boltzmann_constant: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.properties.particle_properties`

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

## particula.gpu.properties.dynamic_viscosity_wp

*function*

```python
def dynamic_viscosity_wp(temperature: wp.float64, ref_viscosity: wp.float64, ref_temperature: wp.float64, sutherland_constant: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.properties.gas_properties`

Calculate dynamic viscosity with Sutherland's formula.

Port of
``particula.gas.properties.dynamic_viscosity.get_dynamic_viscosity``.

**Parameters**

- `temperature` (`wp.float64`) — Gas temperature [K].
- `ref_viscosity` (`wp.float64`) — Reference viscosity at ref_temperature [Pa·s].
- `ref_temperature` (`wp.float64`) — Reference temperature [K].
- `sutherland_constant` (`wp.float64`) — Sutherland constant [K].

**Returns**

- (`wp.float64`) — Dynamic viscosity [Pa·s].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/gas_properties.py#L12-L38)

## particula.gpu.properties.effective_density_wp

*function*

```python
def effective_density_wp(total_mass: wp.float64, total_volume: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.properties.particle_properties`

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

## particula.gpu.properties.friction_factor_wp

*function*

```python
def friction_factor_wp(particle_radius: wp.float64, dynamic_viscosity: wp.float64, slip_correction: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.properties.particle_properties`

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

## particula.gpu.properties.kelvin_radius_wp

*function*

```python
def kelvin_radius_wp(effective_surface_tension: wp.float64, effective_density: wp.float64, molar_mass: wp.float64, temperature: wp.float64, gas_constant: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.properties.particle_properties`

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

## particula.gpu.properties.kelvin_term_wp

*function*

```python
def kelvin_term_wp(particle_radius: wp.float64, kelvin_radius_value: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.properties.particle_properties`

Calculate the Kelvin term with safe clamping.

Port of
``particula.particles.properties.kelvin_effect_module.get_kelvin_term``.

**Parameters**

- `particle_radius` (`wp.float64`) — Particle radius [m].
- `kelvin_radius_value` (`wp.float64`) — Kelvin radius [m].

**Returns**

- (`wp.float64`) — Kelvin term (dimensionless).

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L694-L716)

## particula.gpu.properties.knudsen_number_wp

*function*

```python
def knudsen_number_wp(mean_free_path: wp.float64, particle_radius: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.properties.particle_properties`

Calculate the Knudsen number from mean free path and radius.

Port of ``particula.particles.properties.knudsen_number_module``.

**Parameters**

- `mean_free_path` (`wp.float64`) — Mean free path of the gas molecules [m].
- `particle_radius` (`wp.float64`) — Particle radius [m].

**Returns**

- (`wp.float64`) — Knudsen number (dimensionless).

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L62-L78)

## particula.gpu.properties.mean_thermal_speed_wp

*function*

```python
def mean_thermal_speed_wp(particle_mass: wp.float64, temperature: wp.float64, boltzmann_constant: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.properties.particle_properties`

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

## particula.gpu.properties.molecule_mean_free_path_wp

*function*

```python
def molecule_mean_free_path_wp(molar_mass: wp.float64, temperature: wp.float64, pressure: wp.float64, dynamic_viscosity: wp.float64, gas_constant: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.properties.gas_properties`

Calculate mean free path for a gas molecule.

Port of
``particula.gas.properties.mean_free_path.get_molecule_mean_free_path``.

**Parameters**

- `molar_mass` (`wp.float64`) — Molar mass of the gas [kg/mol].
- `temperature` (`wp.float64`) — Gas temperature [K].
- `pressure` (`wp.float64`) — Gas pressure [Pa].
- `dynamic_viscosity` (`wp.float64`) — Dynamic viscosity [Pa·s].
- `gas_constant` (`wp.float64`) — Universal gas constant [J/(mol·K)].

**Returns**

- (`wp.float64`) — Mean free path of a gas molecule [m].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/gas_properties.py#L41-L67)

## particula.gpu.properties.partial_pressure_delta_wp

*function*

```python
def partial_pressure_delta_wp(partial_pressure_gas: wp.float64, partial_pressure_particle: wp.float64, kelvin_term: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.properties.particle_properties`

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

## particula.gpu.properties.partial_pressure_wp

*function*

```python
def partial_pressure_wp(concentration: wp.float64, molar_mass: wp.float64, temperature: wp.float64, gas_constant: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.properties.gas_properties`

Calculate partial pressure from concentration and temperature.

Port of ``particula.gas.properties.pressure_function.get_partial_pressure``.

**Parameters**

- `concentration` (`wp.float64`) — Gas concentration [kg/m³].
- `molar_mass` (`wp.float64`) — Molar mass [kg/mol].
- `temperature` (`wp.float64`) — Gas temperature [K].
- `gas_constant` (`wp.float64`) — Universal gas constant [J/(mol·K)].

**Returns**

- (`wp.float64`) — Partial pressure [Pa].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/gas_properties.py#L70-L90)

## particula.gpu.properties.particle_radius_from_volume_wp

*function*

```python
def particle_radius_from_volume_wp(total_volume: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.properties.particle_properties`

Calculate a spherical particle radius from total volume.

Evaluates ``r = (3 V / (4 pi))**(1/3)`` for volume ``V``. Non-finite and
non-positive volumes return exact ``0.0`` before the cube root.

**Parameters**

- `total_volume` (`wp.float64`) — Total spherical-equivalent particle volume ``V`` [m³].

**Returns**

- (`wp.float64`) — Spherical-equivalent particle radius [m], or exact ``0.0`` for an
- (`wp.float64`) — invalid volume.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L116-L135)

## particula.gpu.properties.settling_velocity_stokes_from_transport_wp

*function*

```python
def settling_velocity_stokes_from_transport_wp(particle_radius: wp.float64, effective_density: wp.float64, dynamic_viscosity: wp.float64, mean_free_path: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.properties.particle_properties`

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

## particula.gpu.properties.settling_velocity_stokes_wp

*function*

```python
def settling_velocity_stokes_wp(particle_radius: wp.float64, effective_density: wp.float64, temperature: wp.float64, pressure: wp.float64, gas_constant: wp.float64, molecular_weight_air: wp.float64, ref_viscosity: wp.float64, ref_temperature: wp.float64, sutherland_constant: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.properties.particle_properties`

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

## particula.gpu.properties.vapor_transition_correction_wp

*function*

```python
def vapor_transition_correction_wp(knudsen_number: wp.float64, mass_accommodation: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.properties.particle_properties`

Calculate the vapor transition correction factor.

Port of
``particula.particles.properties.vapor_correction_module.get_vapor_transition_correction``.

**Parameters**

- `knudsen_number` (`wp.float64`) — Knudsen number (dimensionless).
- `mass_accommodation` (`wp.float64`) — Mass accommodation coefficient (dimensionless).

**Returns**

- (`wp.float64`) — Vapor transition correction factor (dimensionless).

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/particle_properties.py#L635-L663)

## particula.gpu.properties.x_coth_x_wp

*function*

```python
def x_coth_x_wp(x: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.properties.particle_properties`

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

**Modules**

- `particula.gpu.properties.gas_properties`
- `particula.gpu.properties.particle_properties`
