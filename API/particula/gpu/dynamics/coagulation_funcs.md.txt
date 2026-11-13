# particula.gpu.dynamics.coagulation_funcs

*module*

Warp device functions for Brownian, sedimentation, and charged coagulation.

The Brownian helpers mirror ``particula.dynamics.coagulation.brownian_kernel``.
The internal sedimentation helpers calculate mixture effective density, Stokes
settling velocity with Cunningham slip, and the unit-efficiency SP2016 pair
rate using fp64 scalar device inputs.
The concrete-module-only charged hard-sphere helper supports the low-level,
charged-only and canonical Brownian-plus-charged particle-resolved GPU paths.
It does not add public exports or higher-level charged coagulation APIs.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L1-L1)

## particula.gpu.dynamics.coagulation_funcs._COULOMB_SERIES_THRESHOLD

*attribute* · *module attribute*

```python
_COULOMB_SERIES_THRESHOLD = wp.constant(wp.float64(1e-05))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L27-L27)

## particula.gpu.dynamics.coagulation_funcs._HARD_SPHERE_DENOMINATOR_LINEAR

*attribute* · *module attribute*

```python
_HARD_SPHERE_DENOMINATOR_LINEAR = wp.constant(wp.float64(3.502))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L31-L31)

## particula.gpu.dynamics.coagulation_funcs._HARD_SPHERE_DENOMINATOR_QUADRATIC

*attribute* · *module attribute*

```python
_HARD_SPHERE_DENOMINATOR_QUADRATIC = wp.constant(wp.float64(7.211))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L32-L32)

## particula.gpu.dynamics.coagulation_funcs._HARD_SPHERE_LINEAR

*attribute* · *module attribute*

```python
_HARD_SPHERE_LINEAR = wp.constant(wp.float64(25.836))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L29-L29)

## particula.gpu.dynamics.coagulation_funcs._HARD_SPHERE_QUARTIC

*attribute* · *module attribute*

```python
_HARD_SPHERE_QUARTIC = wp.constant(wp.float64(11.211))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L30-L30)

## particula.gpu.dynamics.coagulation_funcs._KINETIC_LIMIT_CUTOFF

*attribute* · *module attribute*

```python
_KINETIC_LIMIT_CUTOFF = wp.constant(wp.float64(1e-80))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L28-L28)

## particula.gpu.dynamics.coagulation_funcs._MIN_NORMAL_FLOAT64

*attribute* · *module attribute*

```python
_MIN_NORMAL_FLOAT64 = wp.constant(wp.float64(2.2250738585072014e-308))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L26-L26)

## particula.gpu.dynamics.coagulation_funcs._PI

*attribute* · *module attribute*

```python
_PI = wp.constant(wp.float64(3.141592653589793))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L25-L25)

## particula.gpu.dynamics.coagulation_funcs.brownian_diffusivity_wp

*function*

```python
def brownian_diffusivity_wp(temperature: wp.float64, aerodynamic_mobility: wp.float64, boltzmann_constant: wp.float64) -> wp.float64
```

Calculate Brownian diffusivity via Stokes-Einstein scaling.

Port of
``particula.dynamics.coagulation.brownian_kernel._brownian_diffusivity``
using ``D = boltzmann_constant * temperature * aerodynamic_mobility``.

**Parameters**

- `temperature` (`wp.float64`) — Gas temperature in kelvin.
- `aerodynamic_mobility` (`wp.float64`) — Aerodynamic mobility with units consistent with the Stokes-Einstein relation.
- `boltzmann_constant` (`wp.float64`) — Boltzmann constant in joules per kelvin.

**Returns**

- (`wp.float64`) — Brownian diffusivity in square meters per second.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L154-L175)

## particula.gpu.dynamics.coagulation_funcs.brownian_kernel_pair_wp

*function*

```python
def brownian_kernel_pair_wp(radius_i: wp.float64, radius_j: wp.float64, diff_i: wp.float64, diff_j: wp.float64, g_i: wp.float64, g_j: wp.float64, speed_i: wp.float64, speed_j: wp.float64, alpha: wp.float64) -> wp.float64
```

Calculate the scalar Brownian coagulation kernel for a pair.

Port of
``particula.dynamics.coagulation.brownian_kernel.get_brownian_kernel`` for
a scalar particle pair using the Fuchs correction to the continuum kernel.

**Parameters**

- `radius_i` (`wp.float64`) — Particle radius for particle i in meters.
- `radius_j` (`wp.float64`) — Particle radius for particle j in meters.
- `diff_i` (`wp.float64`) — Particle diffusivity for particle i in square meters per second.
- `diff_j` (`wp.float64`) — Particle diffusivity for particle j in square meters per second.
- `g_i` (`wp.float64`) — Collection term for particle i (dimensionless).
- `g_j` (`wp.float64`) — Collection term for particle j (dimensionless).
- `speed_i` (`wp.float64`) — Mean thermal speed for particle i in meters per second.
- `speed_j` (`wp.float64`) — Mean thermal speed for particle j in meters per second.
- `alpha` (`wp.float64`) — Collision efficiency (dimensionless).

**Returns**

- (`wp.float64`) — Brownian coagulation kernel for the pair in cubic meters per second.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L235-L283)

## particula.gpu.dynamics.coagulation_funcs.charged_hard_sphere_wp

*function*

```python
def charged_hard_sphere_wp(radius_i: wp.float64, radius_j: wp.float64, mass_i: wp.float64, mass_j: wp.float64, charge_i: wp.float64, charge_j: wp.float64, temperature: wp.float64, pressure: wp.float64, boltzmann_constant: wp.float64, elementary_charge_value: wp.float64, electric_permittivity: wp.float64, gas_constant: wp.float64, molecular_weight_air: wp.float64, ref_viscosity: wp.float64, ref_temperature: wp.float64, sutherland_constant: wp.float64) -> wp.float64
```

Calculate an internal charged hard-sphere pair rate in SI units.

This device-only, concrete-module-only helper is the pair-rate primitive for
charged-only and canonical Brownian-plus-charged particle-resolved low-level
GPU coagulation. It receives each particle's total mass across species and
is used for candidate rates and compact-active majorants. It ports the
charged CPU calculation from
``particula.dynamics.coagulation.charged_dimensional_kernel`` and the
hard-sphere fit from
``particula.dynamics.coagulation.charged_dimensionless_kernel``. It does
not change public exports.

Non-finite or non-positive radii, masses, thermodynamic state, or physical
constants return exact safe zero. Charges must be finite but may be signed
or zero. Invalid intermediates, finite-input intermediate overflow,
overflow-derived signed zero, and underflow return exact safe zero rather
than a non-finite or negative rate. Finite extreme repulsion is clipped at
a potential ratio of ``-200.0`` and returns safe zero through the kinetic
cutoff when its enhancement is below ``1e-80``.

**Parameters**

- `radius_i` (`wp.float64`) — Radius of particle i [m]. Must be finite and positive.
- `radius_j` (`wp.float64`) — Radius of particle j [m]. Must be finite and positive.
- `mass_i` (`wp.float64`) — Mass of particle i [kg]. Must be finite and positive.
- `mass_j` (`wp.float64`) — Mass of particle j [kg]. Must be finite and positive.
- `charge_i` (`wp.float64`) — Signed charge of particle i in elementary-charge counts.
- `charge_j` (`wp.float64`) — Signed charge of particle j in elementary-charge counts.
- `temperature` (`wp.float64`) — Gas temperature [K]. Must be finite and positive.
- `pressure` (`wp.float64`) — Gas pressure [Pa]. Must be finite and positive.
- `boltzmann_constant` (`wp.float64`) — Boltzmann constant [J/K]. Must be finite and positive.
- `elementary_charge_value` (`wp.float64`) — Elementary-charge magnitude [C]. Must be finite and positive.
- `electric_permittivity` (`wp.float64`) — Vacuum electric permittivity [F/m]. Must be finite and positive.
- `gas_constant` (`wp.float64`) — Universal gas constant [J/(mol K)]. Must be finite and positive.
- `molecular_weight_air` (`wp.float64`) — Molar mass of air [kg/mol]. Must be finite and positive.
- `ref_viscosity` (`wp.float64`) — Reference gas dynamic viscosity [Pa s]. Must be finite and positive.
- `ref_temperature` (`wp.float64`) — Reference temperature for Sutherland viscosity [K]. Must be finite and positive.
- `sutherland_constant` (`wp.float64`) — Sutherland temperature constant [K]. Must be finite and positive.

**Returns**

- (`wp.float64`) — Finite, non-negative charged hard-sphere pair rate [m³/s], or exact
- (`wp.float64`) — ``0.0`` for invalid inputs, invalid or overflowed intermediates,
- (`wp.float64`) — finite extreme-repulsion clipping, or the kinetic cutoff.

> **References**
>
> Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007).
> Coagulation of particles in the transition regime: The effect of the
> Coulomb potential. *Journal of Chemical Physics*, 126(12).
> https://doi.org/10.1063/1.2713719

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L520-L773)

## particula.gpu.dynamics.coagulation_funcs.coulomb_continuum_limit_wp

*function*

```python
def coulomb_continuum_limit_wp(coulomb_potential_ratio: wp.float64) -> wp.float64
```

Calculate the dimensionless continuum Coulomb enhancement factor.

Computes ``phi / (1 - exp(-phi))``. A neutral potential safely returns the
limiting value ``1.0``.

**Parameters**

- `coulomb_potential_ratio` (`wp.float64`) — Dimensionless Coulomb potential ratio.

**Returns**

- (`wp.float64`) — Dimensionless continuum-limit Coulomb enhancement factor.

> **References**
>
> Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced
> collisions in aerosols and dusty plasmas. *Physical Review E*, 85(2).
> https://doi.org/10.1103/PhysRevE.85.026410

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L418-L452)

## particula.gpu.dynamics.coagulation_funcs.coulomb_kinetic_limit_wp

*function*

```python
def coulomb_kinetic_limit_wp(coulomb_potential_ratio: wp.float64) -> wp.float64
```

Calculate the dimensionless kinetic Coulomb enhancement factor.

Computes ``1 + phi`` for ``phi >= 0`` and ``exp(phi)`` otherwise. Finite
scalar inputs have no fallback condition.

**Parameters**

- `coulomb_potential_ratio` (`wp.float64`) — Dimensionless Coulomb potential ratio.

**Returns**

- (`wp.float64`) — Dimensionless kinetic-limit Coulomb enhancement factor.

> **References**
>
> Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced
> collisions in aerosols and dusty plasmas. *Physical Review E*, 85(2).
> https://doi.org/10.1103/PhysRevE.85.026410

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L393-L415)

## particula.gpu.dynamics.coagulation_funcs.coulomb_potential_ratio_wp

*function*

```python
def coulomb_potential_ratio_wp(radius_i: wp.float64, radius_j: wp.float64, charge_i: wp.float64, charge_j: wp.float64, temperature: wp.float64, boltzmann_constant: wp.float64, elementary_charge_value: wp.float64, electric_permittivity: wp.float64) -> wp.float64
```

Calculate the dimensionless pair Coulomb potential ratio.

Computes ``-(q_i * q_j * e**2) / (4 * pi * epsilon_0 * (r_i + r_j)
* k_B * T)``. A non-positive radius sum or temperature, or a non-finite
intermediate from finite extreme inputs, safely returns ``0.0``. Repulsion
is lower-clipped at ``-200.0``.

**Parameters**

- `radius_i` (`wp.float64`) — Particle i radius in meters.
- `radius_j` (`wp.float64`) — Particle j radius in meters.
- `charge_i` (`wp.float64`) — Particle i charge as an elementary-charge count.
- `charge_j` (`wp.float64`) — Particle j charge as an elementary-charge count.
- `temperature` (`wp.float64`) — Gas temperature in kelvin.
- `boltzmann_constant` (`wp.float64`) — Boltzmann constant in joules per kelvin.
- `elementary_charge_value` (`wp.float64`) — Elementary charge magnitude in coulombs.
- `electric_permittivity` (`wp.float64`) — Vacuum electric permittivity in farads per meter.

**Returns**

- (`wp.float64`) — Dimensionless Coulomb potential ratio, or ``0.0`` for an invalid
- (`wp.float64`) — radius sum, temperature, or non-finite intermediate.

> **References**
>
> Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced
> collisions in aerosols and dusty plasmas. *Physical Review E*, 85(2).
> https://doi.org/10.1103/PhysRevE.85.026410

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L286-L361)

## particula.gpu.dynamics.coagulation_funcs.diffusive_knudsen_number_wp

*function*

```python
def diffusive_knudsen_number_wp(radius_i: wp.float64, radius_j: wp.float64, mass_i: wp.float64, mass_j: wp.float64, friction_i: wp.float64, friction_j: wp.float64, coulomb_potential_ratio: wp.float64, temperature: wp.float64, boltzmann_constant: wp.float64) -> wp.float64
```

Calculate the dimensionless pair diffusive Knudsen number.

Computes ``[sqrt(k_B * T * m_red) / f_red] / [(r_i + r_j) * Gamma_c /
Gamma_k]``. A non-positive radius sum, temperature, reduced mass, or
reduced friction safely returns ``0.0``. A kinetic enhancement below
``1e-80`` also returns ``0.0`` for extreme repulsion.

**Parameters**

- `radius_i` (`wp.float64`) — Particle i radius in meters.
- `radius_j` (`wp.float64`) — Particle j radius in meters.
- `mass_i` (`wp.float64`) — Particle i mass in kilograms.
- `mass_j` (`wp.float64`) — Particle j mass in kilograms.
- `friction_i` (`wp.float64`) — Particle i friction coefficient in kilograms per second.
- `friction_j` (`wp.float64`) — Particle j friction coefficient in kilograms per second.
- `coulomb_potential_ratio` (`wp.float64`) — Dimensionless Coulomb potential ratio.
- `temperature` (`wp.float64`) — Gas temperature in kelvin.
- `boltzmann_constant` (`wp.float64`) — Boltzmann constant in joules per kelvin.

**Returns**

- (`wp.float64`) — Dimensionless diffusive Knudsen number, or ``0.0`` for an invalid
- (`wp.float64`) — scalar domain or a kinetic enhancement below ``1e-80``.

> **References**
>
> Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
> molecular regime Coulombic collisions in aerosols and dusty plasmas.
> *Aerosol Science and Technology*, 53(8), 933-957.
> https://doi.org/10.1080/02786826.2019.1614522
> Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced
> collisions in aerosols and dusty plasmas. *Physical Review E*, 85(2).
> https://doi.org/10.1103/PhysRevE.85.026410

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L455-L517)

## particula.gpu.dynamics.coagulation_funcs.g_collection_term_wp

*function*

```python
def g_collection_term_wp(mean_free_path_particle: wp.float64, particle_radius: wp.float64) -> wp.float64
```

Calculate the Brownian coagulation collection term ``g``.

Port of
``particula.dynamics.coagulation.brownian_kernel._g_collection_term`` using
the Fuchs form for the collection enhancement term.

**Parameters**

- `mean_free_path_particle` (`wp.float64`) — Particle mean free path in meters.
- `particle_radius` (`wp.float64`) — Particle radius in meters.

**Returns**

- (`wp.float64`) — Collection term ``g`` as a dimensionless quantity.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L204-L232)

## particula.gpu.dynamics.coagulation_funcs.kinematic_viscosity_wp

*function*

```python
def kinematic_viscosity_wp(dynamic_viscosity: wp.float64, fluid_density: wp.float64) -> wp.float64
```

Calculate kinematic viscosity from device-native SI transport inputs.

This device-only helper applies ``nu = mu / rho`` for positive finite
dynamic viscosity ``mu`` [Pa s] and fluid density ``rho`` [kg/m³].
Public-input validation is intentionally deferred to the calling boundary.

**Parameters**

- `dynamic_viscosity` (`wp.float64`) — Dynamic viscosity ``mu`` [Pa s].
- `fluid_density` (`wp.float64`) — Fluid density ``rho`` [kg/m³].

**Returns**

- (`wp.float64`) — Kinematic viscosity ``nu`` [m²/s].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L90-L108)

## particula.gpu.dynamics.coagulation_funcs.particle_mean_free_path_wp

*function*

```python
def particle_mean_free_path_wp(diffusivity_particle: wp.float64, mean_thermal_speed_particle: wp.float64) -> wp.float64
```

Calculate the particle mean free path for coagulation.

Port of ``particula.dynamics.coagulation.brownian_kernel._mean_free_path_l``
using ``lambda = 8 * diffusivity_particle / (pi * mean_thermal_speed)``.

**Parameters**

- `diffusivity_particle` (`wp.float64`) — Particle diffusivity in square meters per second.
- `mean_thermal_speed_particle` (`wp.float64`) — Particle mean thermal speed in meters per second.

**Returns**

- (`wp.float64`) — Particle mean free path in meters.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L178-L201)

## particula.gpu.dynamics.coagulation_funcs.reduced_value_wp

*function*

```python
def reduced_value_wp(left: wp.float64, right: wp.float64) -> wp.float64
```

Calculate a scalar reduced value for mass or friction inputs.

Computes ``(left * right) / (left + right)``. Inputs retain their supplied
units. A non-positive input sum safely returns ``0.0``.

**Parameters**

- `left` (`wp.float64`) — First mass in kilograms or friction coefficient in kilograms per second.
- `right` (`wp.float64`) — Second mass in kilograms or friction coefficient in kilograms per second.

**Returns**

- (`wp.float64`) — Reduced value in the input units, or ``0.0`` when the input sum is
- (`wp.float64`) — non-positive.

> **References**
>
> Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
> molecular regime Coulombic collisions in aerosols and dusty plasmas.
> *Aerosol Science and Technology*, 53(8), 933-957.
> https://doi.org/10.1080/02786826.2019.1614522

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L364-L390)

## particula.gpu.dynamics.coagulation_funcs.sedimentation_sp2016_pair_rate_wp

*function*

```python
def sedimentation_sp2016_pair_rate_wp(radius_i: wp.float64, radius_j: wp.float64, velocity_i: wp.float64, velocity_j: wp.float64) -> wp.float64
```

Calculate the fp64 SP2016 Eq. 13A.4 sedimentation pair rate.

This device-only helper applies unit collision efficiency to
``K = π (r_i + r_j)² |v_i - v_j|``. Non-finite or negative inputs and
non-finite, overflowed, or underflowed intermediates and results return
exact ``0.0``. Equal finite velocities also return exact ``0.0``.

**Parameters**

- `radius_i` (`wp.float64`) — Particle i radius [m].
- `radius_j` (`wp.float64`) — Particle j radius [m].
- `velocity_i` (`wp.float64`) — Particle i settling velocity [m/s].
- `velocity_j` (`wp.float64`) — Particle j settling velocity [m/s].

**Returns**

- (`wp.float64`) — Sedimentation pair rate [m³/s], or exact safe zero for an invalid or
- (`wp.float64`) — zero-rate calculation.

> **References**
>
> Seinfeld, J. H., & Pandis, S. N. (2016). *Atmospheric Chemistry and
> Physics* (3rd ed.), Eq. 13A.4.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L35-L87)

## particula.gpu.dynamics.coagulation_funcs.turbulent_shear_st1956_pair_rate_wp

*function*

```python
def turbulent_shear_st1956_pair_rate_wp(radius_i: wp.float64, radius_j: wp.float64, turbulent_dissipation: wp.float64, kinematic_viscosity: wp.float64) -> wp.float64
```

Calculate the ST1956 turbulent-shear pair rate in SI units.

This device-only helper applies
``K = sqrt(pi * epsilon / (120 * nu)) * (2 r_i + 2 r_j)**3`` for finite
positive radii ``r`` [m] and kinematic viscosity ``nu`` [m²/s], with finite
non-negative turbulent dissipation ``epsilon`` [m²/s³]. Zero dissipation
returns exact ``0.0`` before the diameter sum, preventing ``0 * inf`` for
finite extreme radii. Callers must preflight derived transport and radius
state; they must not use rate sanitization to conceal invalid physics.

**Parameters**

- `radius_i` (`wp.float64`) — Particle i radius [m].
- `radius_j` (`wp.float64`) — Particle j radius [m].
- `turbulent_dissipation` (`wp.float64`) — Turbulent energy dissipation ``epsilon`` [m²/s³].
- `kinematic_viscosity` (`wp.float64`) — Fluid kinematic viscosity ``nu`` [m²/s].

**Returns**

- (`wp.float64`) — Turbulent-shear pair rate [m³/s].

> **References**
>
> Saffman, P. G., & Turner, J. S. (1956). On the collision of drops in
> turbulent clouds. *Journal of Fluid Mechanics*, 1(1), 16-30.
> Seinfeld, J. H., & Pandis, S. N. (2016). *Atmospheric Chemistry and
> Physics* (3rd ed.), Eq. 13A.2.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L111-L151)
