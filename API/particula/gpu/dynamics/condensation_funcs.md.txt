# particula.gpu.dynamics.condensation_funcs

*module*

Warp GPU condensation composite functions.

These functions mirror NumPy implementations in
``particula.particles.properties`` and
``particula.dynamics.condensation.mass_transfer``. Private thermal helpers
use fp64 non-isothermal equations during each fixed substep in
``particula.gpu.kernels.condensation``. The correction uses the same
activity- and Kelvin-adjusted surface vapor pressure as the mass-transfer
driving pressure; zero latent heat retains the isothermal rate.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/condensation_funcs.py#L1-L1)

## particula.gpu.dynamics.condensation_funcs._PI

*attribute* · *module attribute*

```python
_PI = wp.constant(wp.float64(3.141592653589793))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/condensation_funcs.py#L14-L14)

## particula.gpu.dynamics.condensation_funcs._THERMAL_CONDUCTIVITY_GRADIENT

*attribute* · *module attribute*

```python
_THERMAL_CONDUCTIVITY_GRADIENT = wp.constant(wp.float64(0.071))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/condensation_funcs.py#L17-L17)

## particula.gpu.dynamics.condensation_funcs._THERMAL_CONDUCTIVITY_INTERCEPT

*attribute* · *module attribute*

```python
_THERMAL_CONDUCTIVITY_INTERCEPT = wp.constant(wp.float64(4.39))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/condensation_funcs.py#L16-L16)

## particula.gpu.dynamics.condensation_funcs._THERMAL_CONDUCTIVITY_SCALE

*attribute* · *module attribute*

```python
_THERMAL_CONDUCTIVITY_SCALE = wp.constant(wp.float64(0.001))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/condensation_funcs.py#L15-L15)

## particula.gpu.dynamics.condensation_funcs._mass_transfer_rate_latent_heat_wp

*function*

```python
def _mass_transfer_rate_latent_heat_wp(pressure_delta: wp.float64, first_order_mass_transport: wp.float64, temperature: wp.float64, molar_mass: wp.float64, latent_heat: wp.float64, thermal_conductivity: wp.float64, vapor_pressure_surface: wp.float64, diffusion_coefficient: wp.float64, gas_constant: wp.float64) -> wp.float64
```

Calculate the latent-corrected condensation mass transfer rate.

The non-isothermal correction uses the supplied activity- and
Kelvin-adjusted surface vapor pressure. A zero latent heat returns
``mass_transfer_rate_wp`` exactly, preserving the isothermal result.

**Parameters**

- `pressure_delta` (`wp.float64`) — Gas-to-particle partial-pressure difference [Pa].
- `first_order_mass_transport` (`wp.float64`) — First-order mass transport coefficient [m³/s].
- `temperature` (`wp.float64`) — Gas temperature [K].
- `molar_mass` (`wp.float64`) — Species molar mass [kg/mol].
- `latent_heat` (`wp.float64`) — Species latent heat [J/kg].
- `thermal_conductivity` (`wp.float64`) — Air thermal conductivity [W/(m·K)].
- `vapor_pressure_surface` (`wp.float64`) — Surface vapor pressure [Pa].
- `diffusion_coefficient` (`wp.float64`) — Vapor diffusion coefficient [m²/s].
- `gas_constant` (`wp.float64`) — Universal gas constant [J/(mol·K)].

**Returns**

- (`wp.float64`) — Latent-corrected mass transfer rate [kg/s].

> **References**
>
> Topping, D. O., & Bane, M. K. (2022), Equation 2.36.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/condensation_funcs.py#L138-L212)

## particula.gpu.dynamics.condensation_funcs._thermal_conductivity_wp

*function*

```python
def _thermal_conductivity_wp(temperature: wp.float64) -> wp.float64
```

Calculate air thermal conductivity from temperature.

**Parameters**

- `temperature` (`wp.float64`) — Air temperature [K].

**Returns**

- (`wp.float64`) — Air thermal conductivity [W/(m·K)].

> **References**
>
> Seinfeld, J. H., & Pandis, S. N. (2016). *Atmospheric Chemistry
> and Physics*, Equation 17.54.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/condensation_funcs.py#L79-L96)

## particula.gpu.dynamics.condensation_funcs._thermal_resistance_factor_wp

*function*

```python
def _thermal_resistance_factor_wp(diffusion_coefficient: wp.float64, latent_heat: wp.float64, vapor_pressure_surface: wp.float64, thermal_conductivity: wp.float64, temperature: wp.float64, molar_mass: wp.float64, gas_constant: wp.float64) -> wp.float64
```

Calculate the latent-heat thermal resistance factor.

**Parameters**

- `diffusion_coefficient` (`wp.float64`) — Vapor diffusion coefficient [m²/s].
- `latent_heat` (`wp.float64`) — Species latent heat [J/kg].
- `vapor_pressure_surface` (`wp.float64`) — Surface vapor pressure [Pa].
- `thermal_conductivity` (`wp.float64`) — Air thermal conductivity [W/(m·K)].
- `temperature` (`wp.float64`) — Gas temperature [K].
- `molar_mass` (`wp.float64`) — Species molar mass [kg/mol].
- `gas_constant` (`wp.float64`) — Universal gas constant [J/(mol·K)].

**Returns**

- (`wp.float64`) — Thermal resistance factor [J/kg].

> **References**
>
> Topping, D. O., & Bane, M. K. (2022), Equation 2.36.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/condensation_funcs.py#L99-L135)

## particula.gpu.dynamics.condensation_funcs.effective_surface_tension_wp

*function*

```python
def effective_surface_tension_wp(masses: wp.array(dtype=wp.float64, ndim=3), densities: wp.array(dtype=wp.float64), surface_tensions: wp.array(dtype=wp.float64), box_idx: int, particle_idx: int, requested_species_idx: int, composition_weighted: bool) -> wp.float64
```

Return static or composition-weighted surface tension for one particle.

In static mode, immediately return the requested species tension [N/m]
without reading ``masses`` or ``densities``. In composition-weighted mode,
ignore ``requested_species_idx`` and calculate the single-phase value as
``sum(surface_tension * mass / density) / sum(mass / density)`` across the
fixed ``n_species`` axis. If the total volume is zero, return the
unweighted arithmetic mean of all supplied species tensions instead.

**Parameters**

- `masses` (`wp.array(dtype=wp.float64, ndim=3)`) — Species masses [kg] with fixed shape ``(n_boxes, n_particles, n_species)``.
- `densities` (`wp.array(dtype=wp.float64)`) — Species densities [kg/m³] with fixed shape ``(n_species,)``.
- `surface_tensions` (`wp.array(dtype=wp.float64)`) — Species surface tensions [N/m] with fixed shape ``(n_species,)``.
- `box_idx` (`int`) — Index of the particle box.
- `particle_idx` (`int`) — Index of the particle within ``box_idx``.
- `requested_species_idx` (`int`) — Species index used only in static mode.
- `composition_weighted` (`bool`) — Selects composition-weighted mode when true; otherwise selects static mode.

**Returns**

- (`wp.float64`) — Effective surface tension [N/m].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/condensation_funcs.py#L215-L268)

## particula.gpu.dynamics.condensation_funcs.first_order_mass_transport_k_wp

*function*

```python
def first_order_mass_transport_k_wp(particle_radius: wp.float64, vapor_transition: wp.float64, diffusion_coefficient: wp.float64) -> wp.float64
```

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

## particula.gpu.dynamics.condensation_funcs.mass_transfer_rate_wp

*function*

```python
def mass_transfer_rate_wp(pressure_delta: wp.float64, first_order_mass_transport: wp.float64, temperature: wp.float64, molar_mass: wp.float64, gas_constant: wp.float64) -> wp.float64
```

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

## particula.gpu.dynamics.condensation_funcs.water_activity_ideal_wp

*function*

```python
def water_activity_ideal_wp(masses: wp.array(dtype=wp.float64, ndim=3), molar_masses: wp.array(dtype=wp.float64), box_idx: int, particle_idx: int, water_index: int) -> wp.float64
```

Calculate ideal water activity for one particle from its mole fraction.

This Warp helper mirrors
``particula.particles.properties.activity_module.get_ideal_activity_molar``.
It returns zero when the selected particle has zero total moles.

**Parameters**

- `masses` (`wp.array(dtype=wp.float64, ndim=3)`) — Species masses [kg], shaped ``(n_boxes, n_particles, n_species)``.
- `molar_masses` (`wp.array(dtype=wp.float64)`) — Species molar masses [kg/mol].
- `box_idx` (`int`) — Index of the particle box (dimensionless).
- `particle_idx` (`int`) — Particle index in the box.
- `water_index` (`int`) — Index of the water species (dimensionless).

**Returns**

- (`wp.float64`) — Dimensionless water mole fraction, or zero for zero total moles.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/condensation_funcs.py#L271-L310)

## particula.gpu.dynamics.condensation_funcs.water_activity_kappa_wp

*function*

```python
def water_activity_kappa_wp(masses: wp.array(dtype=wp.float64, ndim=3), densities: wp.array(dtype=wp.float64), kappas: wp.array(dtype=wp.float64), box_idx: int, particle_idx: int, water_index: int) -> wp.float64
```

Calculate κ-model water activity for one particle.

This Warp helper mirrors
``particula.particles.properties.activity_module.get_kappa_activity``.
It returns zero when water volume is zero and one when solute volume is
zero.

**Parameters**

- `masses` (`wp.array(dtype=wp.float64, ndim=3)`) — Species masses [kg], shaped ``(n_boxes, n_particles, n_species)``.
- `densities` (`wp.array(dtype=wp.float64)`) — Species densities [kg/m³].
- `kappas` (`wp.array(dtype=wp.float64)`) — Species hygroscopicity parameters (dimensionless).
- `box_idx` (`int`) — Index of the particle box (dimensionless).
- `particle_idx` (`int`) — Particle index in the box.
- `water_index` (`int`) — Index of the water species (dimensionless).

**Returns**

- (`wp.float64`) — Dimensionless water activity.

> **References**
>
> Petters, M. D., & Kreidenweis, S. M. (2007). A single parameter
> representation of hygroscopic growth and cloud condensation nucleus
> activity. Atmospheric Chemistry and Physics, 7(8), 1961-1971.
> https://doi.org/10.5194/acp-7-1961-2007

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/dynamics/condensation_funcs.py#L313-L367)
