# particula.gpu.dynamics.wall_loss_funcs

*module*

Provide fp64 Warp wall-loss coefficient and private image-charge helpers.

The helpers calculate Crump-Seinfeld spherical and rectangular chamber
wall-loss coefficients in SI units [s^-1], plus private image-charge enhancement
image-charge
enhancement primitives with dimensionless outputs. Coulomb self-potential
ratios are lower-clipped at -200, while image-charge exponents use the CPU
diagonal/self-pair absolute-value calculation and clip to [-50, 50]. Private
Field drift helpers resolve geometry-specific fields, calculate signed drift,
and
sanitize charged coefficients. They are composed by the direct fixed-shape
wall-loss step for nonzero charged slots; zero-charge slots retain its neutral
coefficient and RNG path. The helpers are device-only and have no public
Validation contract; direct-step frozen preflight owns input validation.

Crump, J. G., & Seinfeld, J. H. (1981). Turbulent deposition and
gravitational sedimentation of an aerosol in a vessel of arbitrary shape.
*Journal of Aerosol Science*, 12(5).
https://doi.org/10.1016/0021-8502(81)90036-7

Crump, J. G., Flagan, R. C., & Seinfeld, J. H. (1982). Particle wall loss
rates in vessels. *Aerosol Science and Technology*, 2(3), 303-309.
https://doi.org/10.1080/02786828308958636

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/wall_loss_funcs.py#L1-L1)

## particula.gpu.dynamics.wall_loss_funcs._CHARGED_DRIFT_LOWER_GUARD

*attribute* · *module attribute*

```python
_CHARGED_DRIFT_LOWER_GUARD = wp.constant(wp.float64(1e-30))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/wall_loss_funcs.py#L47-L47)

## particula.gpu.dynamics.wall_loss_funcs._COULOMB_RATIO_LOWER_LIMIT

*attribute* · *module attribute*

```python
_COULOMB_RATIO_LOWER_LIMIT = wp.constant(wp.float64(-200.0))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/wall_loss_funcs.py#L44-L44)

## particula.gpu.dynamics.wall_loss_funcs._FLOAT64_MAX

*attribute* · *module attribute*

```python
_FLOAT64_MAX = wp.constant(wp.float64(1.7976931348623157e+308))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/wall_loss_funcs.py#L48-L48)

## particula.gpu.dynamics.wall_loss_funcs._IMAGE_CHARGE_EXPONENT_LOWER_LIMIT

*attribute* · *module attribute*

```python
_IMAGE_CHARGE_EXPONENT_LOWER_LIMIT = wp.constant(wp.float64(-50.0))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/wall_loss_funcs.py#L45-L45)

## particula.gpu.dynamics.wall_loss_funcs._IMAGE_CHARGE_EXPONENT_UPPER_LIMIT

*attribute* · *module attribute*

```python
_IMAGE_CHARGE_EXPONENT_UPPER_LIMIT = wp.constant(wp.float64(50.0))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/wall_loss_funcs.py#L46-L46)

## particula.gpu.dynamics.wall_loss_funcs._PI

*attribute* · *module attribute*

```python
_PI = wp.constant(wp.float64(3.141592653589793))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/wall_loss_funcs.py#L43-L43)

## particula.gpu.dynamics.wall_loss_funcs._combine_charged_wall_loss_coefficient_wp

*function*

```python
def _combine_charged_wall_loss_coefficient_wp(neutral_coefficient: wp.float64, electrostatic_factor: wp.float64, drift_term: wp.float64) -> wp.float64
```

Combine and sanitize charged wall-loss coefficient contributions.

Calculates ``neutral_coefficient * electrostatic_factor + drift_term``.
NaN and nonpositive results map to zero, while positive overflow and
infinity map to the largest finite float64 value.

**Parameters**

- `neutral_coefficient` (`wp.float64`) — Neutral wall-loss coefficient in s^-1.
- `electrostatic_factor` (`wp.float64`) — Dimensionless image-charge enhancement factor.
- `drift_term` (`wp.float64`) — Signed electric-field drift contribution in s^-1.

**Returns**

- (`wp.float64`) — Finite nonnegative charged wall-loss coefficient in s^-1.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/wall_loss_funcs.py#L544-L569)

## particula.gpu.dynamics.wall_loss_funcs._coulomb_self_potential_ratio_wp

*function*

```python
def _coulomb_self_potential_ratio_wp(particle_radius: wp.float64, particle_charge: wp.float64, temperature: wp.float64, elementary_charge_value: wp.float64, electric_permittivity: wp.float64, boltzmann_constant: wp.float64) -> wp.float64
```

Calculate the dimensionless Coulomb self-potential ratio.

Calculates the particle self-pair ratio and lower-clips it at -200. This
private device helper expects finite physical constants and positive,
finite radius and temperature. It performs no validation; direct-step
preflight owns validation.

**Parameters**

- `particle_radius` (`wp.float64`) — Particle radius in m.
- `particle_charge` (`wp.float64`) — Particle charge in elementary-charge units.
- `temperature` (`wp.float64`) — Gas temperature in K.
- `elementary_charge_value` (`wp.float64`) — Elementary charge in C.
- `electric_permittivity` (`wp.float64`) — Electric permittivity in F/m.
- `boltzmann_constant` (`wp.float64`) — Boltzmann constant in J/K.

**Returns**

- (`wp.float64`) — Dimensionless self-potential ratio, lower-clipped at -200.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/wall_loss_funcs.py#L274-L314)

## particula.gpu.dynamics.wall_loss_funcs._electric_field_drift_from_viscosity_wp

*function*

```python
def _electric_field_drift_from_viscosity_wp(particle_radius: wp.float64, particle_charge: wp.float64, resolved_electric_field: wp.float64, geometry_scale: wp.float64, elementary_charge_value: wp.float64, viscosity: wp.float64) -> wp.float64
```

Calculate signed electric-field drift from precomputed viscosity.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/wall_loss_funcs.py#L510-L541)

## particula.gpu.dynamics.wall_loss_funcs._electric_field_drift_wp

*function*

```python
def _electric_field_drift_wp(particle_radius: wp.float64, particle_charge: wp.float64, temperature: wp.float64, resolved_electric_field: wp.float64, geometry_scale: wp.float64, elementary_charge_value: wp.float64, ref_viscosity: wp.float64, ref_temperature: wp.float64, sutherland_constant: wp.float64) -> wp.float64
```

Calculate signed electric-field drift contribution in s^-1.

Uses Sutherland dynamic viscosity and charge mobility. Particle radius and
the drift denominator are lower-guarded at 1e-30. NaN drift is mapped to
zero while finite signed values and infinities are preserved for final
coefficient composition.

**Parameters**

- `particle_radius` (`wp.float64`) — Particle radius in m.
- `particle_charge` (`wp.float64`) — Particle charge in elementary-charge units.
- `temperature` (`wp.float64`) — Gas temperature in K.
- `resolved_electric_field` (`wp.float64`) — Geometry-resolved electric field in V/m.
- `geometry_scale` (`wp.float64`) — Characteristic chamber scale in m.
- `elementary_charge_value` (`wp.float64`) — Elementary charge in C.
- `ref_viscosity` (`wp.float64`) — Reference dynamic viscosity in Pa s.
- `ref_temperature` (`wp.float64`) — Reference temperature for viscosity in K.
- `sutherland_constant` (`wp.float64`) — Sutherland temperature constant in K.

**Returns**

- (`wp.float64`) — Signed electric-field drift contribution in s^-1.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/wall_loss_funcs.py#L457-L507)

## particula.gpu.dynamics.wall_loss_funcs._geometry_scale_wp

*function*

```python
def _geometry_scale_wp(geometry_mode: wp.int32, chamber_radius: wp.float64, chamber_length: wp.float64, chamber_width: wp.float64, chamber_height: wp.float64) -> wp.float64
```

Return the unguarded characteristic chamber scale in m.

Spherical chambers use their radius; rectangular chambers use their
smallest dimension. Field resolution deliberately uses this unguarded
scale to match the CPU charged-wall-loss strategy.

**Parameters**

- `geometry_mode` (`wp.int32`) — Zero for spherical geometry; nonzero for rectangular.
- `chamber_radius` (`wp.float64`) — Spherical chamber radius in m.
- `chamber_length` (`wp.float64`) — Rectangular chamber length in m.
- `chamber_width` (`wp.float64`) — Rectangular chamber width in m.
- `chamber_height` (`wp.float64`) — Rectangular chamber height in m.

**Returns**

- (`wp.float64`) — Characteristic geometry scale in m without a lower-bound guard.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/wall_loss_funcs.py#L366-L392)

## particula.gpu.dynamics.wall_loss_funcs._image_charge_enhancement_wp

*function*

```python
def _image_charge_enhancement_wp(particle_radius: wp.float64, particle_charge: wp.float64, temperature: wp.float64, elementary_charge_value: wp.float64, electric_permittivity: wp.float64, boltzmann_constant: wp.float64) -> wp.float64
```

Calculate the dimensionless image-charge wall-loss enhancement.

Returns exactly 1.0 for zero charge. Otherwise, lower-clips the Coulomb
self-potential ratio at -200, takes its absolute value to match CPU
diagonal/self-pair physics, clips the exponent to [-50, 50], and
exponentiates it. This private device helper expects finite physical
constants and positive, finite radius and temperature. It performs no
validation; direct-step preflight owns validation.

**Parameters**

- `particle_radius` (`wp.float64`) — Particle radius in m.
- `particle_charge` (`wp.float64`) — Particle charge in elementary-charge units.
- `temperature` (`wp.float64`) — Gas temperature in K.
- `elementary_charge_value` (`wp.float64`) — Elementary charge in C.
- `electric_permittivity` (`wp.float64`) — Electric permittivity in F/m.
- `boltzmann_constant` (`wp.float64`) — Boltzmann constant in J/K.

**Returns**

- (`wp.float64`) — Dimensionless image-charge enhancement factor with a clipped exponent.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/wall_loss_funcs.py#L317-L363)

## particula.gpu.dynamics.wall_loss_funcs._resolve_rectangular_electric_field_wp

*function*

```python
def _resolve_rectangular_electric_field_wp(field_x: wp.float64, field_y: wp.float64, field_z: wp.float64, wall_potential: wp.float64, geometry_scale: wp.float64) -> wp.float64
```

Resolve a signed rectangular electric field in V/m.

Computes the Euclidean magnitude of the three supplied field components,
then conditionally adds the potential-derived field. A signed potential
may therefore decrease or reverse the resolved field.

**Parameters**

- `field_x` (`wp.float64`) — X electric-field component in V/m.
- `field_y` (`wp.float64`) — Y electric-field component in V/m.
- `field_z` (`wp.float64`) — Z electric-field component in V/m.
- `wall_potential` (`wp.float64`) — Wall potential in V.
- `geometry_scale` (`wp.float64`) — Unguarded characteristic chamber scale in m.

**Returns**

- (`wp.float64`) — Signed resolved electric field in V/m.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/wall_loss_funcs.py#L420-L454)

## particula.gpu.dynamics.wall_loss_funcs._resolve_spherical_electric_field_wp

*function*

```python
def _resolve_spherical_electric_field_wp(wall_electric_field: wp.float64, wall_potential: wp.float64, geometry_scale: wp.float64) -> wp.float64
```

Resolve a signed spherical electric field in V/m.

Adds the potential-derived field only for a nonzero potential and positive
geometry scale. The supplied scalar field remains signed.

**Parameters**

- `wall_electric_field` (`wp.float64`) — Signed scalar electric field in V/m.
- `wall_potential` (`wp.float64`) — Wall potential in V.
- `geometry_scale` (`wp.float64`) — Unguarded characteristic chamber scale in m.

**Returns**

- (`wp.float64`) — Signed resolved electric field in V/m.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/wall_loss_funcs.py#L395-L417)

## particula.gpu.dynamics.wall_loss_funcs.rectangle_wall_loss_coefficient_from_transport_wp

*function*

```python
def rectangle_wall_loss_coefficient_from_transport_wp(wall_eddy_diffusivity: wp.float64, particle_radius: wp.float64, particle_density: wp.float64, temperature: wp.float64, chamber_length: wp.float64, chamber_width: wp.float64, chamber_height: wp.float64, dynamic_viscosity: wp.float64, mean_free_path: wp.float64, boltzmann_constant: wp.float64) -> wp.float64
```

Calculate a rectangular coefficient from precomputed gas transport.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/wall_loss_funcs.py#L226-L271)

## particula.gpu.dynamics.wall_loss_funcs.rectangle_wall_loss_coefficient_wp

*function*

```python
def rectangle_wall_loss_coefficient_wp(wall_eddy_diffusivity: wp.float64, particle_radius: wp.float64, particle_density: wp.float64, temperature: wp.float64, pressure: wp.float64, chamber_length: wp.float64, chamber_width: wp.float64, chamber_height: wp.float64, boltzmann_constant: wp.float64, gas_constant: wp.float64, molecular_weight_air: wp.float64, ref_viscosity: wp.float64, ref_temperature: wp.float64, sutherland_constant: wp.float64) -> wp.float64
```

Calculate the neutral rectangular wall-loss coefficient in s^-1.

Combines side-wall turbulent deposition with gravitational sedimentation
using the Crump-Seinfeld rectangular-chamber relation. This concrete Warp
device helper assumes supported physical inputs and performs no public
input validation or charged-particle correction.

**Parameters**

- `wall_eddy_diffusivity` (`wp.float64`) — Wall eddy diffusivity in m^2/s.
- `particle_radius` (`wp.float64`) — Particle radius in m.
- `particle_density` (`wp.float64`) — Particle material density in kg/m^3.
- `temperature` (`wp.float64`) — Gas temperature in K.
- `pressure` (`wp.float64`) — Gas pressure in Pa.
- `chamber_length` (`wp.float64`) — Rectangular chamber length in m.
- `chamber_width` (`wp.float64`) — Rectangular chamber width in m.
- `chamber_height` (`wp.float64`) — Rectangular chamber height in m.
- `boltzmann_constant` (`wp.float64`) — Boltzmann constant in J/K.
- `gas_constant` (`wp.float64`) — Universal gas constant in J/(mol K).
- `molecular_weight_air` (`wp.float64`) — Mean molecular weight of air in kg/mol.
- `ref_viscosity` (`wp.float64`) — Reference dynamic viscosity in Pa s.
- `ref_temperature` (`wp.float64`) — Reference temperature for viscosity in K.
- `sutherland_constant` (`wp.float64`) — Sutherland temperature constant in K.

**Returns**

- (`wp.float64`) — Neutral particle wall-loss coefficient in s^-1.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/wall_loss_funcs.py#L156-L223)

## particula.gpu.dynamics.wall_loss_funcs.spherical_wall_loss_coefficient_from_transport_wp

*function*

```python
def spherical_wall_loss_coefficient_from_transport_wp(wall_eddy_diffusivity: wp.float64, particle_radius: wp.float64, particle_density: wp.float64, temperature: wp.float64, chamber_radius: wp.float64, dynamic_viscosity: wp.float64, mean_free_path: wp.float64, boltzmann_constant: wp.float64) -> wp.float64
```

Calculate a spherical coefficient from precomputed gas transport.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/wall_loss_funcs.py#L115-L153)

## particula.gpu.dynamics.wall_loss_funcs.spherical_wall_loss_coefficient_wp

*function*

```python
def spherical_wall_loss_coefficient_wp(wall_eddy_diffusivity: wp.float64, particle_radius: wp.float64, particle_density: wp.float64, temperature: wp.float64, pressure: wp.float64, chamber_radius: wp.float64, boltzmann_constant: wp.float64, gas_constant: wp.float64, molecular_weight_air: wp.float64, ref_viscosity: wp.float64, ref_temperature: wp.float64, sutherland_constant: wp.float64) -> wp.float64
```

Calculate the neutral spherical wall-loss coefficient in s^-1.

Combines diffusion-driven turbulent deposition with gravitational
sedimentation using the Crump-Seinfeld spherical-chamber relation. This
concrete Warp device helper assumes supported physical inputs and performs
no public input validation or charged-particle correction.

**Parameters**

- `wall_eddy_diffusivity` (`wp.float64`) — Wall eddy diffusivity in m^2/s.
- `particle_radius` (`wp.float64`) — Particle radius in m.
- `particle_density` (`wp.float64`) — Particle material density in kg/m^3.
- `temperature` (`wp.float64`) — Gas temperature in K.
- `pressure` (`wp.float64`) — Gas pressure in Pa.
- `chamber_radius` (`wp.float64`) — Spherical chamber radius in m.
- `boltzmann_constant` (`wp.float64`) — Boltzmann constant in J/K.
- `gas_constant` (`wp.float64`) — Universal gas constant in J/(mol K).
- `molecular_weight_air` (`wp.float64`) — Mean molecular weight of air in kg/mol.
- `ref_viscosity` (`wp.float64`) — Reference dynamic viscosity in Pa s.
- `ref_temperature` (`wp.float64`) — Reference temperature for viscosity in K.
- `sutherland_constant` (`wp.float64`) — Sutherland temperature constant in K.

**Returns**

- (`wp.float64`) — Neutral particle wall-loss coefficient in s^-1.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/wall_loss_funcs.py#L51-L112)
