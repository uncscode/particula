# particula.gpu.dynamics

*module*

Warp GPU dynamics functions.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/__init__.py#L1-L1)

## particula.gpu.dynamics.brownian_diffusivity_wp

*function*

```python
def brownian_diffusivity_wp(temperature: wp.float64, aerodynamic_mobility: wp.float64, boltzmann_constant: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.dynamics.coagulation_funcs`

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

## particula.gpu.dynamics.brownian_kernel_pair_wp

*function*

```python
def brownian_kernel_pair_wp(radius_i: wp.float64, radius_j: wp.float64, diff_i: wp.float64, diff_j: wp.float64, g_i: wp.float64, g_j: wp.float64, speed_i: wp.float64, speed_j: wp.float64, alpha: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.dynamics.coagulation_funcs`

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

## particula.gpu.dynamics.diffusion_coefficient_wp

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

## particula.gpu.dynamics.first_order_mass_transport_k_wp

*function*

```python
def first_order_mass_transport_k_wp(particle_radius: wp.float64, vapor_transition: wp.float64, diffusion_coefficient: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.dynamics.condensation_funcs`

Calculate the first-order mass transport coefficient.

Port of
``particula.dynamics.condensation.mass_transfer.get_first_order_mass_transport_k``.

**Parameters**

- `particle_radius` (`wp.float64`) — Particle radius [m].
- `vapor_transition` (`wp.float64`) — Vapor transition correction factor (dimensionless).
- `diffusion_coefficient` (`wp.float64`) — Diffusion coefficient [m²/s].

**Returns**

- (`wp.float64`) — First-order mass transport coefficient [m³/s].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/condensation_funcs.py#L20-L45)

## particula.gpu.dynamics.g_collection_term_wp

*function*

```python
def g_collection_term_wp(mean_free_path_particle: wp.float64, particle_radius: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.dynamics.coagulation_funcs`

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

## particula.gpu.dynamics.mass_transfer_rate_wp

*function*

```python
def mass_transfer_rate_wp(pressure_delta: wp.float64, first_order_mass_transport: wp.float64, temperature: wp.float64, molar_mass: wp.float64, gas_constant: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.dynamics.condensation_funcs`

Calculate the condensation mass transfer rate.

Port of
``particula.dynamics.condensation.mass_transfer.get_mass_transfer_rate``.

**Parameters**

- `pressure_delta` (`wp.float64`) — Partial pressure difference [Pa].
- `first_order_mass_transport` (`wp.float64`) — Mass transport coefficient [m³/s].
- `temperature` (`wp.float64`) — Temperature [K].
- `molar_mass` (`wp.float64`) — Molar mass [kg/mol].
- `gas_constant` (`wp.float64`) — Universal gas constant [J/(mol·K)].

**Returns**

- (`wp.float64`) — Mass transfer rate [kg/s].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/condensation_funcs.py#L48-L76)

## particula.gpu.dynamics.particle_mean_free_path_wp

*function*

```python
def particle_mean_free_path_wp(diffusivity_particle: wp.float64, mean_thermal_speed_particle: wp.float64) -> wp.float64
```

Re-exported from: `particula.gpu.dynamics.coagulation_funcs`

Calculate the particle mean free path for coagulation.

Port of ``particula.dynamics.coagulation.brownian_kernel._mean_free_path_l``
using ``lambda = 8 * diffusivity_particle / (pi * mean_thermal_speed)``.

**Parameters**

- `diffusivity_particle` (`wp.float64`) — Particle diffusivity in square meters per second.
- `mean_thermal_speed_particle` (`wp.float64`) — Particle mean thermal speed in meters per second.

**Returns**

- (`wp.float64`) — Particle mean free path in meters.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/coagulation_funcs.py#L178-L201)

**Modules**

- `particula.gpu.dynamics.coagulation_funcs`
- `particula.gpu.dynamics.condensation_funcs`
- `particula.gpu.dynamics.wall_loss_funcs`
