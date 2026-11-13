# particula.dynamics.condensation.condensation_strategies

*module*

Provide condensation strategies for aerosol mass transfer.

This module defines the abstract :class:`CondensationStrategy` base class
alongside concrete implementations that compute mass transfer rates for
isothermal and staggered condensation updates. Helpers cover Knudsen-number
calculations, vapor transition corrections, and batch-aware mass adjustments
that keep updates stable when radii approach continuum limits.

The staggered implementation draws on established numerical methods:

1. **Operator Splitting**: LeVeque (2002) — Finite Volume Methods for
   Hyperbolic Problems.
2. **Symplectic Integration**: Hairer, Lubich, & Wanner (2006) — Geometric
   Numerical Integration.
3. **Mass-Conserving Condensation**: Jacobson (1997, 1999) — Analytical
   Predictor of Condensation (APC).
4. **Particle-Resolved Modeling**: Riemer et al. (2009) — PartMC-MOSAIC.

> **References**
>
> - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and
>   Physics: From Air Pollution to Climate Change (3rd ed.). Wiley.
> - Topping, D. & Bane, M. (2022). Introduction to Aerosol Modelling. Wiley.
>   https://doi.org/10.1002/9781119625728

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1-L1)

## particula.dynamics.condensation.condensation_strategies.MIN_PARTICLE_RADIUS_M

*attribute* · *module attribute*

```python
MIN_PARTICLE_RADIUS_M = 1e-10
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L75-L75)

## particula.dynamics.condensation.condensation_strategies.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L71-L71)

## particula.dynamics.condensation.condensation_strategies.CondensationIsothermal

*class*

```python
class CondensationIsothermal(CondensationStrategy)
```

Bases: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Condensation strategy under isothermal conditions.

This class implements the isothermal condensation model, wherein
temperature remains constant during mass transfer. It calculates
condensation rates based on partial pressure differences, using
no latent heat terms.

**Attributes**

- `Inherits` (`attributes from the base CondensationStrategy`) — molar_mass, diffusion_coefficient, etc.

**Functions**

- `- mass_transfer_rate ` — Calculate the mass transfer rate under
isothermal conditions.
- `- rate ` — Get the per-particle condensation rate, accounting for
concentration.
- `- step ` — Advance the condensation state over a given time step.

**Examples**

```py title="Example Usage"
iso_cond = CondensationIsothermal(
    molar_mass=gas_species.molar_mass,
    activity_strategy=activity,
    surface_strategy=surface,
    vapor_pressure_strategy=vapor_pressure,
    species_map=species_map,
)
rate_array = iso_cond.rate(particle, gas_species, 298.15, 101325)
# particle and gas_species are native single-box containers.
```

> **References**
>
> - Aerosol Modeling, Chapter 2, Equation 2.40
> - Topping, D., & Bane, M. (2022). Introduction to Aerosol Modelling
>     (D. Topping & M. Bane, Eds.). Wiley.
>     [DOI](https://doi.org/10.1002/9781119625728)
> - Seinfeld & Pandis, "Atmospheric Chemistry and Physics," 3rd Ed.,
>   Wiley, 2016.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L852-L1053)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal.accommodation_coefficient

*attribute* · *instance attribute*

```python
accommodation_coefficient = accommodation_coefficient
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L230-L230)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal.activity_strategy

*attribute* · *instance attribute*

```python
activity_strategy = activity_strategy
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L233-L233)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal.diffusion_coefficient

*attribute* · *instance attribute*

```python
diffusion_coefficient = diffusion_coefficient
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L229-L229)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal.molar_mass

*attribute* · *instance attribute*

```python
molar_mass = molar_mass
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L228-L228)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal.skip_partitioning_indices

*attribute* · *instance attribute*

```python
skip_partitioning_indices = skip_partitioning_indices
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L232-L232)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal.species_map

*attribute* · *instance attribute*

```python
species_map = species_map
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L236-L236)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal.surface_strategy

*attribute* · *instance attribute*

```python
surface_strategy = surface_strategy
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L234-L234)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal.update_gases

*attribute* · *instance attribute*

```python
update_gases = update_gases
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L231-L231)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal.vapor_pressure_strategy

*attribute* · *instance attribute*

```python
vapor_pressure_strategy = vapor_pressure_strategy
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L235-L235)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal.__init__

*method*

```python
def __init__(molar_mass: Union[float, NDArray[np.float64]], diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-05, accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0, update_gases: bool = True, skip_partitioning_indices: Optional[Sequence[int]] = None, activity_strategy: ActivityStrategy | None = None, surface_strategy: SurfaceStrategy | None = None, vapor_pressure_strategy: VaporPressureStrategy | Sequence[VaporPressureStrategy] | None = None, *, species_map: CondensationSpeciesMap | None = None)
```

Initialize the CondensationIsothermal strategy.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the species [kg/mol].
- `diffusion_coefficient` (`Union[float, NDArray[np.float64]]`) (default: `2e-05`) — Diffusion coefficient [m^2/s].
- `accommodation_coefficient` (`Union[float, NDArray[np.float64]]`) (default: `1.0`) — Mass accommodation coefficient.
- `update_gases` (`bool`) (default: `True`) — Whether to update gas concentrations on update.
- `skip_partitioning_indices` (`Optional[Sequence[int]]`) (default: `None`) — Species indices that should skip partitioning.
- `activity_strategy` (`ActivityStrategy | None`) (default: `None`) — Activity strategy used for ParticleData inputs.
- `surface_strategy` (`SurfaceStrategy | None`) (default: `None`) — Surface strategy used for ParticleData inputs.
- `vapor_pressure_strategy` (`VaporPressureStrategy | Sequence[VaporPressureStrategy] | None`) (default: `None`) — Vapor pressure strategy used for GasData inputs. Native parameters use the complete gas ordering.
- `species_map` (`CondensationSpeciesMap | None`) (default: `None`) — Explicit gas-name provenance and gas-to-particle pairs required for tagged native data. Unmapped lanes do not transfer.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L894-L935)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal.mass_transfer_rate

*method*

```python
def mass_transfer_rate(particle: ParticleData, gas_species: GasData, temperature: float, pressure: float, dynamic_viscosity: Optional[float] = None) -> NDArray[np.float64]
```

Compute native per-particle mass transfer in full material order.

Radii use the minimum calculation radius. Full-gas transport and vapor
parameters are applied through the explicit species map; activity and
surface laws see the complete particle composition. Inactive slots and
unmapped or skipped lanes have zero rates. Nonfinite pressure
differences
are sanitized to zero after physical configuration validation.

**Parameters**

- `particle` (`ParticleData`) — Tagged single-box ParticleData in SI units.
- `gas_species` (`GasData`) — Full ordered single-box GasData.
- `temperature` (`float`) — Positive finite scalar temperature in K.
- `pressure` (`float`) — Positive finite scalar pressure in Pa.
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — Optional positive finite scalar gas viscosity in Pa s.

**Returns**

- (`NDArray[np.float64]`) — Per-particle rates in kg/s, shape (n_particles, n_particle_species).

**Raises**

- `TypeError` — If native containers or explicit process physics are
missing.
- `ValueError` — If state, species mapping or physical inputs are
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L937-L974)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal.rate

*method*

```python
def rate(particle: ParticleData, gas_species: GasData, temperature: float, pressure: float) -> NDArray[np.float64]
```

Compute each node's contribution to population mass-density change.

Resolved counts are divided by representative volume; discrete number
density is used directly, and radius-PDF values receive whole-grid
quadrature weights. Species mapping and skip indices follow full gas
order while output columns follow particle material order.

**Parameters**

- `particle` (`ParticleData`) — Tagged single-box ParticleData in SI units.
- `gas_species` (`GasData`) — Full ordered single-box GasData.
- `temperature` (`float`) — Positive finite scalar temperature in K.
- `pressure` (`float`) — Positive finite scalar pressure in Pa.

**Returns**

- (`NDArray[np.float64]`) — Contributions in kg/m^3/s, shape (n_particles, n_particle_species).

**Raises**

- `TypeError` — If native containers or explicit process physics are
missing.
- `ValueError` — If state, species mapping or physical inputs are
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L976-L1008)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal.step

*method*

```python
def step(particle: ParticleData, gas_species: GasData, temperature: float, pressure: float, time_step: float, dynamic_viscosity: Optional[float] = None) -> Tuple[ParticleData, GasData]
```

Advance native particle and gas state by one finite timestep.

The step stages inventory-limited particle and gas updates and publishes
only after successful validation and calculation. Gas updates can be
disabled to retain an external reservoir.

**Parameters**

- `particle` (`ParticleData`) — Tagged single-box ParticleData to update in place.
- `gas_species` (`GasData`) — Full ordered single-box GasData.
- `temperature` (`float`) — Positive finite scalar temperature in K.
- `pressure` (`float`) — Positive finite scalar pressure in Pa.
- `time_step` (`float`) — Finite nonnegative duration in seconds.
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — Optional positive finite scalar gas viscosity in Pa s.

**Returns**

- (`ParticleData`) — The identical particle and gas containers, retaining destination
- (`GasData`) — arrays.

**Raises**

- `TypeError` — If native containers or explicit process physics are
missing.
- `ValueError` — If state, storage, configuration or a proposal is
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1012-L1053)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal._native_latent_heat_values

*method*

```python
def _native_latent_heat_values(gas: GasData, temperature: float) -> NDArray[np.float64] | None
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Validate optional latent heat in the complete gas ordering.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L458-L472)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal._native_mass_transfer_rate

*method*

```python
def _native_mass_transfer_rate(particle: ParticleData, gas_species: GasData, temperature: float, pressure: float, dynamic_viscosity: float | None = None) -> NDArray[np.float64]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Return native per-particle kg/s in full particle species order.

Transport and vapor parameters follow the full gas ordering. Activity
and surface strategies see the full particle composition. Pair-specific
scalar transport calls avoid accidental particle/species broadcasting.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L353-L390)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal._native_parameters

*method*

```python
def _native_parameters(particle: ParticleData, gas: GasData, temperature: float, pressure: float) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Admit current state and full-gas transport parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L238-L279)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal._native_pressure_terms

*method*

```python
def _native_pressure_terms(particle: ParticleData, gas: GasData, temperature: float, radius: NDArray[np.float64], sources: NDArray[np.intp], targets: NDArray[np.intp]) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Evaluate full composition and mapped pressure lanes in Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L281-L351)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal._native_rate_from_parameters

*method*

```python
def _native_rate_from_parameters(particle_data: ParticleData, gas: GasData, temperature: float, pressure: float, parameters: tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], radius: NDArray[np.float64], dynamic_viscosity: float | None) -> NDArray[np.float64]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Evaluate admitted material lanes without rescanning a population.

Internal batch material carriers may be untagged: population weights
have already been admitted on the complete grid and are passed to the
inventory finalizer separately. This helper never interprets their
concentration as density or performs public process admission.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L392-L456)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal._native_single_step

*method*

```python
def _native_single_step(particle: ParticleData, gas: GasData, temperature: float, pressure: float, time_step: float, dynamic_viscosity: float | None) -> tuple[ParticleData, GasData]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Stage one mapped isothermal/latent update, then commit by identity.

A disabled gas update keeps the caller's reservoir fixed; the same
finite-stage inventory limiter still bounds the particle proposal.
Latent energy is extensive J, including resolved volume or PDF
quadrature, and follows actual representable particle mass changes.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L474-L537)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal._resolve_strategies

*method*

```python
def _resolve_strategies() -> tuple[ActivityStrategy, SurfaceStrategy, VaporPressureStrategy | Sequence[VaporPressureStrategy]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Require explicit process-owned activity, surface and vapor laws.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L539-L561)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal.calculate_pressure_delta

*method*

```python
def calculate_pressure_delta(particle: ParticleData, gas_species: GasData, temperature: float, radius: NDArray[np.float64]) -> NDArray[np.float64]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Calculate the difference in partial pressure between the gas and
particle phases.

**Parameters**

- `particle` (`ParticleData`) — The tagged single-box ParticleData for which the partial pressure difference is to be calculated.
- `gas_species` (`GasData`) — The full ordered single-box GasData in contact with the particle.
- `temperature` (`float`) — The temperature at which the partial pressure difference is to be calculated.
- `radius` (`NDArray[np.float64]`) — The radius of the particles.

**Returns**

- `partial_pressure_delta` (`NDArray[np.float64]`) — The difference in partial pressure between the gas and particle phases.

**Examples**

```py title="Example – Δp calculation"
delta_p = cond.calculate_pressure_delta(
    particle=particle,
    gas_species=gas_species,
    temperature=298.15,
    radius=particle.radii[0],
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L697-L743)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal.first_order_mass_transport

*method*

```python
def first_order_mass_transport(particle_radius: Union[float, NDArray[np.float64]], temperature: float, pressure: float, dynamic_viscosity: Optional[float] = None) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

First-order mass transport coefficient per particle.

Calculate the first-order mass transport coefficient, K, for a given
particle based on the diffusion coefficient, radius, and vapor
transition correction factor.

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particle [m].
- `temperature` (`float`) — The temperature at which the first-order mass transport coefficient is to be calculated.
- `pressure` (`float`) — The pressure of the gas phase.
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — The dynamic viscosity of the gas [Pa*s]. If not provided, it will be calculated based on the temperature

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The first-order mass transport coefficient per particle (m^3/s).

References:
- Chapter 2, Equation 2.49 (excluding particle number)
- Topping, D., & Bane, M. (2022). Introduction to Aerosol Modelling
    (D. Topping & M. Bane, Eds.). Wiley.
    [DOI](https://doi.org/10.1002/9781119625728)

**Examples**

```py title="Example – First-order mass-transport"
cond = CondensationIsothermal(molar_mass=0.018)
k = cond.first_order_mass_transport(
    particle_radius=1e-7,
    temperature=298.15,
    pressure=101325,
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L642-L695)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal.knudsen_number

*method*

```python
def knudsen_number(radius: Union[float, NDArray[np.float64]], temperature: float, pressure: float, dynamic_viscosity: Optional[float] = None) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

The Knudsen number for a particle.

Calculate the Knudsen number based on the mean free path of the gas
molecules and the radius of the particle.

**Parameters**

- `radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particle [m].
- `temperature` (`float`) — The temperature of the gas [K].
- `pressure` (`float`) — The pressure of the gas [Pa].
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — The dynamic viscosity of the gas [Pa*s]. If not provided, it will be calculated based on the temperature

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The Knudsen number, which is the ratio of the mean free path to the particle radius.

> **References**
>
> - [Knudsen Number](https://en.wikipedia.org/wiki/Knudsen_number)

**Examples**

```py title="Example – Knudsen number"
cond = CondensationIsothermal(molar_mass=0.018)
kn = cond.knudsen_number(
    radius=1e-7, temperature=298.15, pressure=101325
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L599-L640)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermal.mean_free_path

*method*

```python
def mean_free_path(temperature: float, pressure: float, dynamic_viscosity: Optional[float] = None) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Calculate the mean free path of the gas molecules based on the
temperature, pressure, and dynamic viscosity of the gas.

**Parameters**

- `temperature` (`float`) — The temperature of the gas [K].
- `pressure` (`float`) — The pressure of the gas [Pa].
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — The dynamic viscosity of the gas [Pa*s]. If not provided, it will be calculated based on the temperature

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The mean free path of the gas molecules in meters (m).

References:
- Mean Free Path
    [Wikipedia](https://en.wikipedia.org/wiki/Mean_free_path)

**Examples**

```py title="Example – Mean-free-path"
cond = CondensationIsothermal(molar_mass=0.018)  # water vapour
lam = cond.mean_free_path(temperature=298.15, pressure=101325)
print(f"{lam:.2e} m")
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L563-L597)

## particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered

*class*

```python
class CondensationIsothermalStaggered(CondensationStrategy)
```

Bases: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Staggered isothermal condensation with two-pass Gauss-Seidel updates.

This strategy splits each timestep into two passes with inventory-limited
updates for explicitly tagged particle populations. Each pass
executes Gauss-Seidel sweeps over batches of particles and optionally
updates the shared gas field after every batch to reduce lag.

theta_mode controls how the first-pass fraction (theta) is selected:
    - ``"half"`` uses a deterministic half-step (theta = 0.5) for all
      particles.
    - ``"random"`` samples theta ~ U[0, 1] with the configured RNG.
    - ``"batch"`` fixes theta = 1.0 and staggers updates through batches.

**Attributes**

- `theta_mode` — Staggered stepping mode, one of ``("half", "random", "batch")``.
- `num_batches` — Requested Gauss-Seidel batch count clipped to the particle total to avoid empty batches.
- `shuffle_each_step` — When True, particle order is reshuffled each step; set False for deterministic sequencing.
- `random_state` — Optional seed, ``numpy.random.Generator``, or ``RandomState`` reused for shuffling and theta draws to keep the stochastic steps reproducible.

> **Example**
>
> >>> from particula.dynamics.condensation import (
> ...     CondensationIsothermalStaggered,
> ... )
> >>> strategy = CondensationIsothermalStaggered(
> ...     molar_mass=gas_species.molar_mass,
> ...     activity_strategy=activity,
> ...     surface_strategy=surface,
> ...     vapor_pressure_strategy=vapor_pressure,
> ...     species_map=species_map,
> ...     theta_mode="random",
> ...     num_batches=8,
> ...     shuffle_each_step=True,
> ...     random_state=1234,
> ... )
> >>> particle, gas = strategy.step(
> ...     particle, gas_species, 298.0, 101325.0, 0.1
> ... )

> **See Also**
>
> CondensationIsothermal: Simultaneous (non-staggered) condensation.
> CondensationIsothermalStaggeredBuilder: Fluent builder for this
>     strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1056-L1525)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.VALID_THETA_MODES

*attribute* · *class attribute* · *instance attribute*

```python
VALID_THETA_MODES = ('half', 'random', 'batch')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1106-L1106)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.num_batches

*attribute* · *instance attribute*

```python
num_batches = num_batches
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1182-L1182)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.random_state

*attribute* · *instance attribute*

```python
random_state = random_state
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1184-L1184)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.shuffle_each_step

*attribute* · *instance attribute*

```python
shuffle_each_step = shuffle_each_step
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1183-L1183)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.theta_mode

*attribute* · *instance attribute*

```python
theta_mode = theta_mode
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1181-L1181)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.accommodation_coefficient

*attribute* · *instance attribute*

```python
accommodation_coefficient = accommodation_coefficient
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L230-L230)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.activity_strategy

*attribute* · *instance attribute*

```python
activity_strategy = activity_strategy
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L233-L233)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.diffusion_coefficient

*attribute* · *instance attribute*

```python
diffusion_coefficient = diffusion_coefficient
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L229-L229)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.molar_mass

*attribute* · *instance attribute*

```python
molar_mass = molar_mass
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L228-L228)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.skip_partitioning_indices

*attribute* · *instance attribute*

```python
skip_partitioning_indices = skip_partitioning_indices
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L232-L232)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.species_map

*attribute* · *instance attribute*

```python
species_map = species_map
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L236-L236)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.surface_strategy

*attribute* · *instance attribute*

```python
surface_strategy = surface_strategy
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L234-L234)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.update_gases

*attribute* · *instance attribute*

```python
update_gases = update_gases
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L231-L231)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.vapor_pressure_strategy

*attribute* · *instance attribute*

```python
vapor_pressure_strategy = vapor_pressure_strategy
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L235-L235)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.__init__

*method*

```python
def __init__(molar_mass: Union[float, NDArray[np.float64]], theta_mode: str = 'half', num_batches: int = 1, shuffle_each_step: bool = True, random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]] = None, diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-05, accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0, update_gases: bool = True, skip_partitioning_indices: Optional[Sequence[int]] = None, activity_strategy: ActivityStrategy | None = None, surface_strategy: SurfaceStrategy | None = None, vapor_pressure_strategy: VaporPressureStrategy | Sequence[VaporPressureStrategy] | None = None, *, species_map: CondensationSpeciesMap | None = None)
```

Initialize the staggered condensation strategy.

This constructor configures theta mode, batching, and RNG options while
reusing diffusion and accommodation settings from the base strategy. The
requested batch count is clipped to the particle inventory to prevent
empty Gauss-Seidel sweeps.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the condensing species [kg/mol].
- `theta_mode` (`str`) (default: `'half'`) — Staggered stepping mode; one of ``("half", "random", "batch")``.
- `num_batches` (`int`) (default: `1`) — Requested Gauss-Seidel batch count (>= 1).
- `shuffle_each_step` (`bool`) (default: `True`) — When True, shuffle particle order each step; set False for deterministic sequencing.
- `random_state` (`Optional[Union[int, np.random.Generator, np.random.RandomState]]`) (default: `None`) — Optional seed, ``numpy.random.Generator``, or ``RandomState`` reused for shuffling and theta draws.
- `diffusion_coefficient` (`Union[float, NDArray[np.float64]]`) (default: `2e-05`) — Diffusion coefficient [m^2/s].
- `accommodation_coefficient` (`Union[float, NDArray[np.float64]]`) (default: `1.0`) — Mass accommodation coefficient.
- `update_gases` (`bool`) (default: `True`) — Whether to update gas concentrations.
- `skip_partitioning_indices` (`Optional[Sequence[int]]`) (default: `None`) — Species indices to skip partitioning.
- `activity_strategy` (`ActivityStrategy | None`) (default: `None`) — Activity strategy used for ParticleData inputs.
- `surface_strategy` (`SurfaceStrategy | None`) (default: `None`) — Surface strategy used for ParticleData inputs.
- `vapor_pressure_strategy` (`VaporPressureStrategy | Sequence[VaporPressureStrategy] | None`) (default: `None`) — Vapor pressure strategy used for GasData inputs. Native parameters use the complete gas ordering.
- `species_map` (`CondensationSpeciesMap | None`) (default: `None`) — Explicit gas-name provenance and gas-to-particle pairs required for tagged native data. Unmapped lanes do not transfer.

**Raises**

- `ValueError` — If ``theta_mode`` is unsupported or ``num_batches`` is
less than 1.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1109-L1184)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered._get_theta_values

*method*

```python
def _get_theta_values(n_particles: int) -> NDArray[np.float64]
```

Generate first-pass theta fractions for each particle.

Theta splits the timestep into theta and (1 - theta) passes, controlling
how much work occurs during the first Gauss-Seidel sweep. The
configured RNG determines theta draws when ``theta_mode`` is
``"random"``.

**Parameters**

- `n_particles` (`int`) — Number of particles requiring theta fractions.

**Returns**

- (`NDArray[np.float64]`) — Array of shape ``(n_particles,)``: - ``"half"`` returns 0.5 for every particle. - ``"random"`` draws from ``U[0, 1]`` with the configured RNG   or seed so the samples remain reproducible. - ``"batch"`` returns 1.0 because batching provides the   staggering instead of theta.

**Raises**

- `ValueError` — When ``theta_mode`` is unsupported.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1186-L1226)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered._make_batches

*method*

```python
def _make_batches(n_particles: int) -> list[NDArray[np.intp]]
```

Partition particle indices into Gauss-Seidel batches.

**Parameters**

- `n_particles` (`int`) — Number of particles in the simulation.

**Returns**

- (`list[NDArray[np.intp]]`) — List of arrays, one per batch, or an empty list when ``n_particles``
- (`list[NDArray[np.intp]]`) — is zero.

> **Notes**
>
> Clips and logs when ``num_batches`` exceeds ``n_particles`` to avoid
> empty batches while preserving all particles. Shuffling uses the
> configured seed, ``Generator``, or ``RandomState`` when
> ``shuffle_each_step`` is True; order is preserved otherwise.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1260-L1294)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered._native_staggered_step

*method*

```python
def _native_staggered_step(particle: ParticleData, gas: GasData, temperature: float, pressure: float, time_step: float, dynamic_viscosity: float | None) -> tuple[ParticleData, GasData]
```

Apply two conservative Gauss-Seidel sweeps to private working state.

Preserve the existing frozen particle composition, radii and transport
over both sweeps. Gas pressure reads the latest batch inventory, while
evaporation is bounded by actual remaining staged particle mass.
Local theta durations are applied per node, including mixed durations
in a vectorized batch. PDF weights come from the entire original grid.
A disabled gas update retains fixed reservoir pressure between batches.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1322-L1406)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered._validate_native_staggered_controls

*method*

```python
def _validate_native_staggered_controls() -> None
```

Revalidate mutable controls before drawing randomness or writing.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1296-L1320)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered._validate_num_batches

*method*

```python
def _validate_num_batches(num_batches: int, n_particles: int) -> int
```

Ensure batch count respects available particles.

**Parameters**

- `num_batches` (`int`) — Requested number of Gauss-Seidel batches.
- `n_particles` (`int`) — Number of particles available for batching.

**Returns**

- (`int`) — Adjusted batch count clipped to ``n_particles`` when the request
- (`int`) — exceeds the available particles; logs the clipping action.

**Raises**

- `ValueError` — When ``num_batches`` is less than 1.

> **Notes**
>
> The caller skips this helper when ``n_particles`` is zero, so no
> logging occurs for the empty case.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1228-L1258)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.mass_transfer_rate

*method*

```python
def mass_transfer_rate(particle: ParticleData, gas_species: GasData, temperature: float, pressure: float, dynamic_viscosity: Optional[float] = None) -> NDArray[np.float64]
```

Compute native per-particle mass transfer in full material order.

Radii use the minimum calculation radius. Full-gas transport and vapor
parameters are applied through the explicit species map; activity and
surface laws see the complete particle composition. Inactive slots and
unmapped or skipped lanes have zero rates. Nonfinite pressure
differences
are sanitized to zero after physical configuration validation.

**Parameters**

- `particle` (`ParticleData`) — Tagged single-box ParticleData in SI units.
- `gas_species` (`GasData`) — Full ordered single-box GasData.
- `temperature` (`float`) — Positive finite scalar temperature in K.
- `pressure` (`float`) — Positive finite scalar pressure in Pa.
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — Optional positive finite scalar gas viscosity in Pa s.

**Returns**

- (`NDArray[np.float64]`) — Per-particle rates in kg/s, shape (n_particles, n_particle_species).

**Raises**

- `TypeError` — If native containers or explicit process physics are
missing.
- `ValueError` — If state, species mapping or physical inputs are
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1409-L1445)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.rate

*method*

```python
def rate(particle: ParticleData, gas_species: GasData, temperature: float, pressure: float) -> NDArray[np.float64]
```

Compute each node's contribution to population mass-density change.

Resolved counts are divided by representative volume; discrete number
density is used directly, and radius-PDF values receive whole-grid
quadrature weights. Species mapping and skip indices follow full gas
order while output columns follow particle material order.

**Parameters**

- `particle` (`ParticleData`) — Tagged single-box ParticleData in SI units.
- `gas_species` (`GasData`) — Full ordered single-box GasData.
- `temperature` (`float`) — Positive finite scalar temperature in K.
- `pressure` (`float`) — Positive finite scalar pressure in Pa.

**Returns**

- (`NDArray[np.float64]`) — Contributions in kg/m^3/s, shape (n_particles, n_particle_species).

**Raises**

- `TypeError` — If native containers or explicit process physics are
missing.
- `ValueError` — If state, species mapping or physical inputs are
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1447-L1479)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.step

*method*

```python
def step(particle: ParticleData, gas_species: GasData, temperature: float, pressure: float, time_step: float, dynamic_viscosity: Optional[float] = None) -> Tuple[ParticleData, GasData]
```

Advance native particle and gas state by one finite timestep.

Two theta sweeps evaluate vectorized batches with sequential gas
coupling. Material radii/composition are frozen for rate evaluation;
remaining staged mass and current gas limit each proposal. Array writes
publish after staging; consumed random draws are not rolled back.

**Parameters**

- `particle` (`ParticleData`) — Tagged single-box ParticleData to update in place.
- `gas_species` (`GasData`) — Full ordered single-box GasData.
- `temperature` (`float`) — Positive finite scalar temperature in K.
- `pressure` (`float`) — Positive finite scalar pressure in Pa.
- `time_step` (`float`) — Finite nonnegative duration in seconds.
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — Optional positive finite scalar gas viscosity in Pa s.

**Returns**

- (`ParticleData`) — The identical particle and gas containers, retaining destination
- (`GasData`) — arrays.

**Raises**

- `TypeError` — If native containers or explicit process physics are
missing.
- `ValueError` — If state, storage, configuration or a proposal is
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1483-L1525)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered._native_latent_heat_values

*method*

```python
def _native_latent_heat_values(gas: GasData, temperature: float) -> NDArray[np.float64] | None
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Validate optional latent heat in the complete gas ordering.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L458-L472)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered._native_mass_transfer_rate

*method*

```python
def _native_mass_transfer_rate(particle: ParticleData, gas_species: GasData, temperature: float, pressure: float, dynamic_viscosity: float | None = None) -> NDArray[np.float64]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Return native per-particle kg/s in full particle species order.

Transport and vapor parameters follow the full gas ordering. Activity
and surface strategies see the full particle composition. Pair-specific
scalar transport calls avoid accidental particle/species broadcasting.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L353-L390)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered._native_parameters

*method*

```python
def _native_parameters(particle: ParticleData, gas: GasData, temperature: float, pressure: float) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Admit current state and full-gas transport parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L238-L279)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered._native_pressure_terms

*method*

```python
def _native_pressure_terms(particle: ParticleData, gas: GasData, temperature: float, radius: NDArray[np.float64], sources: NDArray[np.intp], targets: NDArray[np.intp]) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Evaluate full composition and mapped pressure lanes in Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L281-L351)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered._native_rate_from_parameters

*method*

```python
def _native_rate_from_parameters(particle_data: ParticleData, gas: GasData, temperature: float, pressure: float, parameters: tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], radius: NDArray[np.float64], dynamic_viscosity: float | None) -> NDArray[np.float64]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Evaluate admitted material lanes without rescanning a population.

Internal batch material carriers may be untagged: population weights
have already been admitted on the complete grid and are passed to the
inventory finalizer separately. This helper never interprets their
concentration as density or performs public process admission.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L392-L456)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered._native_single_step

*method*

```python
def _native_single_step(particle: ParticleData, gas: GasData, temperature: float, pressure: float, time_step: float, dynamic_viscosity: float | None) -> tuple[ParticleData, GasData]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Stage one mapped isothermal/latent update, then commit by identity.

A disabled gas update keeps the caller's reservoir fixed; the same
finite-stage inventory limiter still bounds the particle proposal.
Latent energy is extensive J, including resolved volume or PDF
quadrature, and follows actual representable particle mass changes.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L474-L537)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered._resolve_strategies

*method*

```python
def _resolve_strategies() -> tuple[ActivityStrategy, SurfaceStrategy, VaporPressureStrategy | Sequence[VaporPressureStrategy]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Require explicit process-owned activity, surface and vapor laws.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L539-L561)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.calculate_pressure_delta

*method*

```python
def calculate_pressure_delta(particle: ParticleData, gas_species: GasData, temperature: float, radius: NDArray[np.float64]) -> NDArray[np.float64]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Calculate the difference in partial pressure between the gas and
particle phases.

**Parameters**

- `particle` (`ParticleData`) — The tagged single-box ParticleData for which the partial pressure difference is to be calculated.
- `gas_species` (`GasData`) — The full ordered single-box GasData in contact with the particle.
- `temperature` (`float`) — The temperature at which the partial pressure difference is to be calculated.
- `radius` (`NDArray[np.float64]`) — The radius of the particles.

**Returns**

- `partial_pressure_delta` (`NDArray[np.float64]`) — The difference in partial pressure between the gas and particle phases.

**Examples**

```py title="Example – Δp calculation"
delta_p = cond.calculate_pressure_delta(
    particle=particle,
    gas_species=gas_species,
    temperature=298.15,
    radius=particle.radii[0],
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L697-L743)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.first_order_mass_transport

*method*

```python
def first_order_mass_transport(particle_radius: Union[float, NDArray[np.float64]], temperature: float, pressure: float, dynamic_viscosity: Optional[float] = None) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

First-order mass transport coefficient per particle.

Calculate the first-order mass transport coefficient, K, for a given
particle based on the diffusion coefficient, radius, and vapor
transition correction factor.

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particle [m].
- `temperature` (`float`) — The temperature at which the first-order mass transport coefficient is to be calculated.
- `pressure` (`float`) — The pressure of the gas phase.
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — The dynamic viscosity of the gas [Pa*s]. If not provided, it will be calculated based on the temperature

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The first-order mass transport coefficient per particle (m^3/s).

References:
- Chapter 2, Equation 2.49 (excluding particle number)
- Topping, D., & Bane, M. (2022). Introduction to Aerosol Modelling
    (D. Topping & M. Bane, Eds.). Wiley.
    [DOI](https://doi.org/10.1002/9781119625728)

**Examples**

```py title="Example – First-order mass-transport"
cond = CondensationIsothermal(molar_mass=0.018)
k = cond.first_order_mass_transport(
    particle_radius=1e-7,
    temperature=298.15,
    pressure=101325,
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L642-L695)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.knudsen_number

*method*

```python
def knudsen_number(radius: Union[float, NDArray[np.float64]], temperature: float, pressure: float, dynamic_viscosity: Optional[float] = None) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

The Knudsen number for a particle.

Calculate the Knudsen number based on the mean free path of the gas
molecules and the radius of the particle.

**Parameters**

- `radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particle [m].
- `temperature` (`float`) — The temperature of the gas [K].
- `pressure` (`float`) — The pressure of the gas [Pa].
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — The dynamic viscosity of the gas [Pa*s]. If not provided, it will be calculated based on the temperature

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The Knudsen number, which is the ratio of the mean free path to the particle radius.

> **References**
>
> - [Knudsen Number](https://en.wikipedia.org/wiki/Knudsen_number)

**Examples**

```py title="Example – Knudsen number"
cond = CondensationIsothermal(molar_mass=0.018)
kn = cond.knudsen_number(
    radius=1e-7, temperature=298.15, pressure=101325
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L599-L640)

### particula.dynamics.condensation.condensation_strategies.CondensationIsothermalStaggered.mean_free_path

*method*

```python
def mean_free_path(temperature: float, pressure: float, dynamic_viscosity: Optional[float] = None) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Calculate the mean free path of the gas molecules based on the
temperature, pressure, and dynamic viscosity of the gas.

**Parameters**

- `temperature` (`float`) — The temperature of the gas [K].
- `pressure` (`float`) — The pressure of the gas [Pa].
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — The dynamic viscosity of the gas [Pa*s]. If not provided, it will be calculated based on the temperature

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The mean free path of the gas molecules in meters (m).

References:
- Mean Free Path
    [Wikipedia](https://en.wikipedia.org/wiki/Mean_free_path)

**Examples**

```py title="Example – Mean-free-path"
cond = CondensationIsothermal(molar_mass=0.018)  # water vapour
lam = cond.mean_free_path(temperature=298.15, pressure=101325)
print(f"{lam:.2e} m")
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L563-L597)

## particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat

*class*

```python
class CondensationLatentHeat(CondensationStrategy)
```

Bases: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Condensation strategy with optional latent-heat corrections.

This strategy mirrors the isothermal workflow while optionally applying a
latent-heat correction to the mass-transfer rate. When a latent heat
strategy is supplied, the correction uses thermal resistance and the vapor
pressure at the particle surface. Without a strategy, the behavior matches
the isothermal implementation. Negative or array-like latent heat inputs
log warnings and fall back to the isothermal path.

**Attributes**

- `latent_heat_strategy_input` — Strategy input provided at initialization.
- `latent_heat_input` — Raw latent heat value provided at initialization.
- `last_latent_heat_energy` — Total latent heat energy released in the most recent step [J]. Positive values indicate condensation and negative values indicate evaporation. Overwritten each call.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1528-L1757)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat._latent_heat_strategy

*attribute* · *instance attribute*

```python
_latent_heat_strategy = self._resolve_latent_heat_strategy(latent_heat_strategy=latent_heat_strategy, latent_heat=latent_heat)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1595-L1598)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat.last_latent_heat_energy

*attribute* · *instance attribute*

```python
last_latent_heat_energy = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1599-L1599)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat.latent_heat_input

*attribute* · *instance attribute*

```python
latent_heat_input = latent_heat
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1594-L1594)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat.latent_heat_strategy_input

*attribute* · *instance attribute*

```python
latent_heat_strategy_input = latent_heat_strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1593-L1593)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat.accommodation_coefficient

*attribute* · *instance attribute*

```python
accommodation_coefficient = accommodation_coefficient
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L230-L230)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat.activity_strategy

*attribute* · *instance attribute*

```python
activity_strategy = activity_strategy
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L233-L233)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat.diffusion_coefficient

*attribute* · *instance attribute*

```python
diffusion_coefficient = diffusion_coefficient
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L229-L229)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat.molar_mass

*attribute* · *instance attribute*

```python
molar_mass = molar_mass
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L228-L228)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat.skip_partitioning_indices

*attribute* · *instance attribute*

```python
skip_partitioning_indices = skip_partitioning_indices
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L232-L232)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat.species_map

*attribute* · *instance attribute*

```python
species_map = species_map
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L236-L236)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat.surface_strategy

*attribute* · *instance attribute*

```python
surface_strategy = surface_strategy
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L234-L234)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat.update_gases

*attribute* · *instance attribute*

```python
update_gases = update_gases
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L231-L231)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat.vapor_pressure_strategy

*attribute* · *instance attribute*

```python
vapor_pressure_strategy = vapor_pressure_strategy
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L235-L235)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat.__init__

*method*

```python
def __init__(molar_mass: Union[float, NDArray[np.float64]], diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-05, accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0, update_gases: bool = True, skip_partitioning_indices: Optional[Sequence[int]] = None, activity_strategy: ActivityStrategy | None = None, surface_strategy: SurfaceStrategy | None = None, vapor_pressure_strategy: VaporPressureStrategy | Sequence[VaporPressureStrategy] | None = None, *, species_map: CondensationSpeciesMap | None = None, latent_heat_strategy: LatentHeatStrategy | None = None, latent_heat: float | NDArray[np.float64] = 0.0)
```

Initialize the CondensationLatentHeat strategy.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the species [kg/mol].
- `diffusion_coefficient` (`Union[float, NDArray[np.float64]]`) (default: `2e-05`) — Diffusion coefficient [m^2/s].
- `accommodation_coefficient` (`Union[float, NDArray[np.float64]]`) (default: `1.0`) — Mass accommodation coefficient.
- `update_gases` (`bool`) (default: `True`) — Whether to update gas concentrations on update.
- `skip_partitioning_indices` (`Optional[Sequence[int]]`) (default: `None`) — Species indices that should skip partitioning.
- `activity_strategy` (`ActivityStrategy | None`) (default: `None`) — Activity strategy used for ParticleData inputs.
- `surface_strategy` (`SurfaceStrategy | None`) (default: `None`) — Surface strategy used for ParticleData inputs.
- `vapor_pressure_strategy` (`VaporPressureStrategy | Sequence[VaporPressureStrategy] | None`) (default: `None`) — Vapor pressure strategy used for GasData inputs. Native parameters use the complete gas ordering.
- `species_map` (`CondensationSpeciesMap | None`) (default: `None`) — Explicit gas-name provenance and gas-to-particle pairs required for tagged native data. Unmapped lanes do not transfer.
- `latent_heat_strategy` (`LatentHeatStrategy | None`) (default: `None`) — Optional latent heat strategy to use.
- `latent_heat` (`float | NDArray[np.float64]`) (default: `0.0`) — Scalar latent heat fallback [J/kg].

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1547-L1599)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat._resolve_latent_heat_strategy

*method*

```python
def _resolve_latent_heat_strategy(latent_heat_strategy: LatentHeatStrategy | None, latent_heat: float | NDArray[np.float64]) -> LatentHeatStrategy | None
```

Resolve the latent heat strategy from inputs.

Prefers an explicit strategy, otherwise converts scalar latent heat
values into a constant strategy. Array-like or negative latent heat
values log a warning and fall back to isothermal behavior.

**Parameters**

- `latent_heat_strategy` (`LatentHeatStrategy | None`) — Optional strategy to use directly.
- `latent_heat` (`float | NDArray[np.float64]`) — Scalar or array-like latent heat input [J/kg].

**Returns**

- (`LatentHeatStrategy | None`) — Resolved latent heat strategy, or None for isothermal fallback.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1601-L1638)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat.mass_transfer_rate

*method*

```python
def mass_transfer_rate(particle: ParticleData, gas_species: GasData, temperature: float, pressure: float, dynamic_viscosity: Optional[float] = None) -> NDArray[np.float64]
```

Compute native per-particle mass transfer in full material order.

Radii use the minimum calculation radius. Full-gas transport and vapor
parameters are applied through the explicit species map; activity and
surface laws see the complete particle composition. Inactive slots and
unmapped or skipped lanes have zero rates. Nonfinite pressure
differences
are sanitized to zero after physical configuration validation.

**Parameters**

- `particle` (`ParticleData`) — Tagged single-box ParticleData in SI units.
- `gas_species` (`GasData`) — Full ordered single-box GasData.
- `temperature` (`float`) — Positive finite scalar temperature in K.
- `pressure` (`float`) — Positive finite scalar pressure in Pa.
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — Optional positive finite scalar gas viscosity in Pa s.

**Returns**

- (`NDArray[np.float64]`) — Per-particle rates in kg/s, shape (n_particles, n_particle_species).

**Raises**

- `TypeError` — If native containers or explicit process physics are
missing.
- `ValueError` — If state, species mapping or physical inputs are
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1640-L1676)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat.rate

*method*

```python
def rate(particle: ParticleData, gas_species: GasData, temperature: float, pressure: float) -> NDArray[np.float64]
```

Compute each node's contribution to population mass-density change.

Resolved counts are divided by representative volume; discrete number
density is used directly, and radius-PDF values receive whole-grid
quadrature weights. Species mapping and skip indices follow full gas
order while output columns follow particle material order.

**Parameters**

- `particle` (`ParticleData`) — Tagged single-box ParticleData in SI units.
- `gas_species` (`GasData`) — Full ordered single-box GasData.
- `temperature` (`float`) — Positive finite scalar temperature in K.
- `pressure` (`float`) — Positive finite scalar pressure in Pa.

**Returns**

- (`NDArray[np.float64]`) — Contributions in kg/m^3/s, shape (n_particles, n_particle_species).

**Raises**

- `TypeError` — If native containers or explicit process physics are
missing.
- `ValueError` — If state, species mapping or physical inputs are
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1678-L1710)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat.step

*method*

```python
def step(particle: ParticleData, gas_species: GasData, temperature: float, pressure: float, time_step: float, dynamic_viscosity: Optional[float] = None) -> Tuple[ParticleData, GasData]
```

Advance native particle and gas state by one finite timestep.

The step stages inventory-limited particle and gas updates and publishes
only after successful validation and calculation. Gas updates can be
disabled to retain an external reservoir.
Latent energy is extensive J from actual applied mass changes, positive
for condensation and negative for evaporation; each call overwrites it.

**Parameters**

- `particle` (`ParticleData`) — Tagged single-box ParticleData to update in place.
- `gas_species` (`GasData`) — Full ordered single-box GasData.
- `temperature` (`float`) — Positive finite scalar temperature in K.
- `pressure` (`float`) — Positive finite scalar pressure in Pa.
- `time_step` (`float`) — Finite nonnegative duration in seconds.
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — Optional positive finite scalar gas viscosity in Pa s.

**Returns**

- (`ParticleData`) — The identical particle and gas containers, retaining destination
- (`GasData`) — arrays.

**Raises**

- `TypeError` — If native containers or explicit process physics are
missing.
- `ValueError` — If state, storage, configuration or a proposal is
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1714-L1757)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat._native_latent_heat_values

*method*

```python
def _native_latent_heat_values(gas: GasData, temperature: float) -> NDArray[np.float64] | None
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Validate optional latent heat in the complete gas ordering.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L458-L472)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat._native_mass_transfer_rate

*method*

```python
def _native_mass_transfer_rate(particle: ParticleData, gas_species: GasData, temperature: float, pressure: float, dynamic_viscosity: float | None = None) -> NDArray[np.float64]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Return native per-particle kg/s in full particle species order.

Transport and vapor parameters follow the full gas ordering. Activity
and surface strategies see the full particle composition. Pair-specific
scalar transport calls avoid accidental particle/species broadcasting.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L353-L390)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat._native_parameters

*method*

```python
def _native_parameters(particle: ParticleData, gas: GasData, temperature: float, pressure: float) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Admit current state and full-gas transport parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L238-L279)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat._native_pressure_terms

*method*

```python
def _native_pressure_terms(particle: ParticleData, gas: GasData, temperature: float, radius: NDArray[np.float64], sources: NDArray[np.intp], targets: NDArray[np.intp]) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Evaluate full composition and mapped pressure lanes in Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L281-L351)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat._native_rate_from_parameters

*method*

```python
def _native_rate_from_parameters(particle_data: ParticleData, gas: GasData, temperature: float, pressure: float, parameters: tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], radius: NDArray[np.float64], dynamic_viscosity: float | None) -> NDArray[np.float64]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Evaluate admitted material lanes without rescanning a population.

Internal batch material carriers may be untagged: population weights
have already been admitted on the complete grid and are passed to the
inventory finalizer separately. This helper never interprets their
concentration as density or performs public process admission.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L392-L456)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat._native_single_step

*method*

```python
def _native_single_step(particle: ParticleData, gas: GasData, temperature: float, pressure: float, time_step: float, dynamic_viscosity: float | None) -> tuple[ParticleData, GasData]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Stage one mapped isothermal/latent update, then commit by identity.

A disabled gas update keeps the caller's reservoir fixed; the same
finite-stage inventory limiter still bounds the particle proposal.
Latent energy is extensive J, including resolved volume or PDF
quadrature, and follows actual representable particle mass changes.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L474-L537)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat._resolve_strategies

*method*

```python
def _resolve_strategies() -> tuple[ActivityStrategy, SurfaceStrategy, VaporPressureStrategy | Sequence[VaporPressureStrategy]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Require explicit process-owned activity, surface and vapor laws.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L539-L561)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat.calculate_pressure_delta

*method*

```python
def calculate_pressure_delta(particle: ParticleData, gas_species: GasData, temperature: float, radius: NDArray[np.float64]) -> NDArray[np.float64]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Calculate the difference in partial pressure between the gas and
particle phases.

**Parameters**

- `particle` (`ParticleData`) — The tagged single-box ParticleData for which the partial pressure difference is to be calculated.
- `gas_species` (`GasData`) — The full ordered single-box GasData in contact with the particle.
- `temperature` (`float`) — The temperature at which the partial pressure difference is to be calculated.
- `radius` (`NDArray[np.float64]`) — The radius of the particles.

**Returns**

- `partial_pressure_delta` (`NDArray[np.float64]`) — The difference in partial pressure between the gas and particle phases.

**Examples**

```py title="Example – Δp calculation"
delta_p = cond.calculate_pressure_delta(
    particle=particle,
    gas_species=gas_species,
    temperature=298.15,
    radius=particle.radii[0],
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L697-L743)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat.first_order_mass_transport

*method*

```python
def first_order_mass_transport(particle_radius: Union[float, NDArray[np.float64]], temperature: float, pressure: float, dynamic_viscosity: Optional[float] = None) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

First-order mass transport coefficient per particle.

Calculate the first-order mass transport coefficient, K, for a given
particle based on the diffusion coefficient, radius, and vapor
transition correction factor.

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particle [m].
- `temperature` (`float`) — The temperature at which the first-order mass transport coefficient is to be calculated.
- `pressure` (`float`) — The pressure of the gas phase.
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — The dynamic viscosity of the gas [Pa*s]. If not provided, it will be calculated based on the temperature

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The first-order mass transport coefficient per particle (m^3/s).

References:
- Chapter 2, Equation 2.49 (excluding particle number)
- Topping, D., & Bane, M. (2022). Introduction to Aerosol Modelling
    (D. Topping & M. Bane, Eds.). Wiley.
    [DOI](https://doi.org/10.1002/9781119625728)

**Examples**

```py title="Example – First-order mass-transport"
cond = CondensationIsothermal(molar_mass=0.018)
k = cond.first_order_mass_transport(
    particle_radius=1e-7,
    temperature=298.15,
    pressure=101325,
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L642-L695)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat.knudsen_number

*method*

```python
def knudsen_number(radius: Union[float, NDArray[np.float64]], temperature: float, pressure: float, dynamic_viscosity: Optional[float] = None) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

The Knudsen number for a particle.

Calculate the Knudsen number based on the mean free path of the gas
molecules and the radius of the particle.

**Parameters**

- `radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particle [m].
- `temperature` (`float`) — The temperature of the gas [K].
- `pressure` (`float`) — The pressure of the gas [Pa].
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — The dynamic viscosity of the gas [Pa*s]. If not provided, it will be calculated based on the temperature

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The Knudsen number, which is the ratio of the mean free path to the particle radius.

> **References**
>
> - [Knudsen Number](https://en.wikipedia.org/wiki/Knudsen_number)

**Examples**

```py title="Example – Knudsen number"
cond = CondensationIsothermal(molar_mass=0.018)
kn = cond.knudsen_number(
    radius=1e-7, temperature=298.15, pressure=101325
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L599-L640)

### particula.dynamics.condensation.condensation_strategies.CondensationLatentHeat.mean_free_path

*method*

```python
def mean_free_path(temperature: float, pressure: float, dynamic_viscosity: Optional[float] = None) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Calculate the mean free path of the gas molecules based on the
temperature, pressure, and dynamic viscosity of the gas.

**Parameters**

- `temperature` (`float`) — The temperature of the gas [K].
- `pressure` (`float`) — The pressure of the gas [Pa].
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — The dynamic viscosity of the gas [Pa*s]. If not provided, it will be calculated based on the temperature

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The mean free path of the gas molecules in meters (m).

References:
- Mean Free Path
    [Wikipedia](https://en.wikipedia.org/wiki/Mean_free_path)

**Examples**

```py title="Example – Mean-free-path"
cond = CondensationIsothermal(molar_mass=0.018)  # water vapour
lam = cond.mean_free_path(temperature=298.15, pressure=101325)
print(f"{lam:.2e} m")
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L563-L597)

## particula.dynamics.condensation.condensation_strategies.CondensationStrategy

*class*

```python
class CondensationStrategy(ABC)
```

Bases: `ABC`

Abstract base class for condensation strategies.

This class defines the interface for various condensation models
used in atmospheric physics. Subclasses should implement specific
condensation algorithms based on different physical models and equations.

**Attributes**

- `molar_mass` — The molar mass of the species [kg/mol].
- `diffusion_coefficient` — The diffusion coefficient [m^2/s].
- `accommodation_coefficient` — The mass accommodation coefficient (unitless).
- `update_gases` — Whether to update gas concentrations after condensation.
- `skip_partitioning_indices` — Optional list of indices for species that should not partition during condensation (default is None).

Methods:
- mean_free_path : Calculate the mean free path of the gas molecules.
- knudsen_number : Compute the Knudsen number for a given particle radius.
- first_order_mass_transport : Calculate first-order mass transport
    coefficient.
- calculate_pressure_delta : Compute the partial pressure difference.
- mass_transfer_rate : Abstract method for the mass transfer rate [kg/s].
- rate : Abstract method for condensation rate per particle/bin.
- step : Abstract method to perform one timestep of condensation.

Concrete strategies require native containers and explicit process-owned
activity, surface, vapor-pressure physics and species mapping. Scalar
transport helpers may be evaluated independently of population execution.

References:
- Seinfeld, J. H. & Pandis, S. N. (2016). Atmospheric Chemistry and
  Physics: From Air Pollution to Climate Change (3rd ed.). Wiley.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L148-L848)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy.accommodation_coefficient

*attribute* · *instance attribute*

```python
accommodation_coefficient = accommodation_coefficient
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L230-L230)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy.activity_strategy

*attribute* · *instance attribute*

```python
activity_strategy = activity_strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L233-L233)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy.diffusion_coefficient

*attribute* · *instance attribute*

```python
diffusion_coefficient = diffusion_coefficient
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L229-L229)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy.molar_mass

*attribute* · *instance attribute*

```python
molar_mass = molar_mass
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L228-L228)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy.skip_partitioning_indices

*attribute* · *instance attribute*

```python
skip_partitioning_indices = skip_partitioning_indices
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L232-L232)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy.species_map

*attribute* · *instance attribute*

```python
species_map = species_map
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L236-L236)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy.surface_strategy

*attribute* · *instance attribute*

```python
surface_strategy = surface_strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L234-L234)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy.update_gases

*attribute* · *instance attribute*

```python
update_gases = update_gases
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L231-L231)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy.vapor_pressure_strategy

*attribute* · *instance attribute*

```python
vapor_pressure_strategy = vapor_pressure_strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L235-L235)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy.__init__

*method*

```python
def __init__(molar_mass: Union[float, NDArray[np.float64]], diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-05, accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0, update_gases: bool = True, skip_partitioning_indices: Optional[Sequence[int]] = None, activity_strategy: ActivityStrategy | None = None, surface_strategy: SurfaceStrategy | None = None, vapor_pressure_strategy: VaporPressureStrategy | Sequence[VaporPressureStrategy] | None = None, *, species_map: CondensationSpeciesMap | None = None)
```

Initialize the CondensationStrategy instance.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the species [kg/mol].
- `diffusion_coefficient` (`Union[float, NDArray[np.float64]]`) (default: `2e-05`) — Diffusion coefficient [m^2/s].
- `accommodation_coefficient` (`Union[float, NDArray[np.float64]]`) (default: `1.0`) — Mass accommodation coefficient (unitless).
- `update_gases` (`bool`) (default: `True`) — Flag indicating whether gas concentrations should be updated on condensation.
- `skip_partitioning_indices` (`Optional[Sequence[int]]`) (default: `None`) — Optional list of indices for species that should not partition during condensation (default is None).
- `activity_strategy` (`ActivityStrategy | None`) (default: `None`) — Activity strategy used when ParticleData inputs are provided (required for data-only usage).
- `surface_strategy` (`SurfaceStrategy | None`) (default: `None`) — Surface strategy used when ParticleData inputs are provided (required for data-only usage).
- `vapor_pressure_strategy` (`VaporPressureStrategy | Sequence[VaporPressureStrategy] | None`) (default: `None`) — Vapor pressure strategy or per-species strategies in complete gas order for native inputs.
- `species_map` (`CondensationSpeciesMap | None`) (default: `None`) — Explicit gas-name provenance and gas-to-particle pairs required for tagged native data. Unmapped lanes do not transfer.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L185-L236)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy._native_latent_heat_values

*method*

```python
def _native_latent_heat_values(gas: GasData, temperature: float) -> NDArray[np.float64] | None
```

Validate optional latent heat in the complete gas ordering.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L458-L472)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy._native_mass_transfer_rate

*method*

```python
def _native_mass_transfer_rate(particle: ParticleData, gas_species: GasData, temperature: float, pressure: float, dynamic_viscosity: float | None = None) -> NDArray[np.float64]
```

Return native per-particle kg/s in full particle species order.

Transport and vapor parameters follow the full gas ordering. Activity
and surface strategies see the full particle composition. Pair-specific
scalar transport calls avoid accidental particle/species broadcasting.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L353-L390)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy._native_parameters

*method*

```python
def _native_parameters(particle: ParticleData, gas: GasData, temperature: float, pressure: float) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
```

Admit current state and full-gas transport parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L238-L279)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy._native_pressure_terms

*method*

```python
def _native_pressure_terms(particle: ParticleData, gas: GasData, temperature: float, radius: NDArray[np.float64], sources: NDArray[np.intp], targets: NDArray[np.intp]) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Evaluate full composition and mapped pressure lanes in Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L281-L351)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy._native_rate_from_parameters

*method*

```python
def _native_rate_from_parameters(particle_data: ParticleData, gas: GasData, temperature: float, pressure: float, parameters: tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], radius: NDArray[np.float64], dynamic_viscosity: float | None) -> NDArray[np.float64]
```

Evaluate admitted material lanes without rescanning a population.

Internal batch material carriers may be untagged: population weights
have already been admitted on the complete grid and are passed to the
inventory finalizer separately. This helper never interprets their
concentration as density or performs public process admission.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L392-L456)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy._native_single_step

*method*

```python
def _native_single_step(particle: ParticleData, gas: GasData, temperature: float, pressure: float, time_step: float, dynamic_viscosity: float | None) -> tuple[ParticleData, GasData]
```

Stage one mapped isothermal/latent update, then commit by identity.

A disabled gas update keeps the caller's reservoir fixed; the same
finite-stage inventory limiter still bounds the particle proposal.
Latent energy is extensive J, including resolved volume or PDF
quadrature, and follows actual representable particle mass changes.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L474-L537)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy._resolve_strategies

*method*

```python
def _resolve_strategies() -> tuple[ActivityStrategy, SurfaceStrategy, VaporPressureStrategy | Sequence[VaporPressureStrategy]]
```

Require explicit process-owned activity, surface and vapor laws.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L539-L561)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy.calculate_pressure_delta

*method*

```python
def calculate_pressure_delta(particle: ParticleData, gas_species: GasData, temperature: float, radius: NDArray[np.float64]) -> NDArray[np.float64]
```

Calculate the difference in partial pressure between the gas and
particle phases.

**Parameters**

- `particle` (`ParticleData`) — The tagged single-box ParticleData for which the partial pressure difference is to be calculated.
- `gas_species` (`GasData`) — The full ordered single-box GasData in contact with the particle.
- `temperature` (`float`) — The temperature at which the partial pressure difference is to be calculated.
- `radius` (`NDArray[np.float64]`) — The radius of the particles.

**Returns**

- `partial_pressure_delta` (`NDArray[np.float64]`) — The difference in partial pressure between the gas and particle phases.

**Examples**

```py title="Example – Δp calculation"
delta_p = cond.calculate_pressure_delta(
    particle=particle,
    gas_species=gas_species,
    temperature=298.15,
    radius=particle.radii[0],
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L697-L743)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy.first_order_mass_transport

*method*

```python
def first_order_mass_transport(particle_radius: Union[float, NDArray[np.float64]], temperature: float, pressure: float, dynamic_viscosity: Optional[float] = None) -> Union[float, NDArray[np.float64]]
```

First-order mass transport coefficient per particle.

Calculate the first-order mass transport coefficient, K, for a given
particle based on the diffusion coefficient, radius, and vapor
transition correction factor.

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particle [m].
- `temperature` (`float`) — The temperature at which the first-order mass transport coefficient is to be calculated.
- `pressure` (`float`) — The pressure of the gas phase.
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — The dynamic viscosity of the gas [Pa*s]. If not provided, it will be calculated based on the temperature

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The first-order mass transport coefficient per particle (m^3/s).

References:
- Chapter 2, Equation 2.49 (excluding particle number)
- Topping, D., & Bane, M. (2022). Introduction to Aerosol Modelling
    (D. Topping & M. Bane, Eds.). Wiley.
    [DOI](https://doi.org/10.1002/9781119625728)

**Examples**

```py title="Example – First-order mass-transport"
cond = CondensationIsothermal(molar_mass=0.018)
k = cond.first_order_mass_transport(
    particle_radius=1e-7,
    temperature=298.15,
    pressure=101325,
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L642-L695)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy.knudsen_number

*method*

```python
def knudsen_number(radius: Union[float, NDArray[np.float64]], temperature: float, pressure: float, dynamic_viscosity: Optional[float] = None) -> Union[float, NDArray[np.float64]]
```

The Knudsen number for a particle.

Calculate the Knudsen number based on the mean free path of the gas
molecules and the radius of the particle.

**Parameters**

- `radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particle [m].
- `temperature` (`float`) — The temperature of the gas [K].
- `pressure` (`float`) — The pressure of the gas [Pa].
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — The dynamic viscosity of the gas [Pa*s]. If not provided, it will be calculated based on the temperature

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The Knudsen number, which is the ratio of the mean free path to the particle radius.

> **References**
>
> - [Knudsen Number](https://en.wikipedia.org/wiki/Knudsen_number)

**Examples**

```py title="Example – Knudsen number"
cond = CondensationIsothermal(molar_mass=0.018)
kn = cond.knudsen_number(
    radius=1e-7, temperature=298.15, pressure=101325
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L599-L640)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy.mass_transfer_rate

*method* · *abstract*

```python
def mass_transfer_rate(particle: ParticleData, gas_species: GasData, temperature: float, pressure: float, dynamic_viscosity: Optional[float] = None) -> NDArray[np.float64]
```

Compute native per-particle mass transfer in full material order.

Radii use the minimum calculation radius. Full-gas transport and vapor
parameters are applied through the explicit species map; activity and
surface laws see the complete particle composition. Inactive slots and
unmapped or skipped lanes have zero rates. Nonfinite pressure
differences
are sanitized to zero after physical configuration validation.

**Parameters**

- `particle` (`ParticleData`) — Tagged single-box ParticleData in SI units.
- `gas_species` (`GasData`) — Full ordered single-box GasData.
- `temperature` (`float`) — Positive finite scalar temperature in K.
- `pressure` (`float`) — Positive finite scalar pressure in Pa.
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — Optional positive finite scalar gas viscosity in Pa s.

**Returns**

- (`NDArray[np.float64]`) — Per-particle rates in kg/s, shape (n_particles, n_particle_species).

**Raises**

- `TypeError` — If native containers or explicit process physics are
missing.
- `ValueError` — If state, species mapping or physical inputs are
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L745-L780)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy.mean_free_path

*method*

```python
def mean_free_path(temperature: float, pressure: float, dynamic_viscosity: Optional[float] = None) -> Union[float, NDArray[np.float64]]
```

Calculate the mean free path of the gas molecules based on the
temperature, pressure, and dynamic viscosity of the gas.

**Parameters**

- `temperature` (`float`) — The temperature of the gas [K].
- `pressure` (`float`) — The pressure of the gas [Pa].
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — The dynamic viscosity of the gas [Pa*s]. If not provided, it will be calculated based on the temperature

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The mean free path of the gas molecules in meters (m).

References:
- Mean Free Path
    [Wikipedia](https://en.wikipedia.org/wiki/Mean_free_path)

**Examples**

```py title="Example – Mean-free-path"
cond = CondensationIsothermal(molar_mass=0.018)  # water vapour
lam = cond.mean_free_path(temperature=298.15, pressure=101325)
print(f"{lam:.2e} m")
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L563-L597)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy.rate

*method* · *abstract*

```python
def rate(particle: ParticleData, gas_species: GasData, temperature: float, pressure: float) -> NDArray[np.float64]
```

Compute each node's contribution to population mass-density change.

Resolved counts are divided by representative volume; discrete number
density is used directly, and radius-PDF values receive whole-grid
quadrature weights. Species mapping and skip indices follow full gas
order while output columns follow particle material order.

**Parameters**

- `particle` (`ParticleData`) — Tagged single-box ParticleData in SI units.
- `gas_species` (`GasData`) — Full ordered single-box GasData.
- `temperature` (`float`) — Positive finite scalar temperature in K.
- `pressure` (`float`) — Positive finite scalar pressure in Pa.

**Returns**

- (`NDArray[np.float64]`) — Contributions in kg/m^3/s, shape (n_particles, n_particle_species).

**Raises**

- `TypeError` — If native containers or explicit process physics are
missing.
- `ValueError` — If state, species mapping or physical inputs are
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L782-L811)

### particula.dynamics.condensation.condensation_strategies.CondensationStrategy.step

*method* · *abstract*

```python
def step(particle: ParticleData, gas_species: GasData, temperature: float, pressure: float, time_step: float, dynamic_viscosity: Optional[float] = None) -> Tuple[ParticleData, GasData]
```

Advance native particle and gas state by one finite timestep.

The step stages inventory-limited particle and gas updates and publishes
only after successful validation and calculation. Gas updates can be
disabled to retain an external reservoir.

**Parameters**

- `particle` (`ParticleData`) — Tagged single-box ParticleData to update in place.
- `gas_species` (`GasData`) — Full ordered single-box GasData.
- `temperature` (`float`) — Positive finite scalar temperature in K.
- `pressure` (`float`) — Positive finite scalar pressure in Pa.
- `time_step` (`float`) — Finite nonnegative duration in seconds.
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — Optional positive finite scalar gas viscosity in Pa s.

**Returns**

- (`ParticleData`) — The identical particle and gas containers, retaining destination
- (`GasData`) — arrays.

**Raises**

- `TypeError` — If native containers or explicit process physics are
missing.
- `ValueError` — If state, storage, configuration or a proposal is
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L814-L848)

## particula.dynamics.condensation.condensation_strategies._native_kelvin_lane

*function*

```python
def _native_kelvin_lane(kelvin: NDArray[np.float64], particle: ParticleData, target: int) -> NDArray[np.float64]
```

Select a declared particle lane from scalar or resolved Kelvin terms.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L128-L144)

## particula.dynamics.condensation.condensation_strategies._partial_pressure_from_strategy

*function*

```python
def _partial_pressure_from_strategy(vapor_pressure_strategy: VaporPressureStrategy | Sequence[VaporPressureStrategy], concentration: NDArray[np.float64], molar_mass: NDArray[np.float64], temperature: float) -> NDArray[np.float64]
```

Compute partial pressure for single or per-species strategies.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L98-L125)

## particula.dynamics.condensation.condensation_strategies._pure_vapor_pressure_from_strategy

*function*

```python
def _pure_vapor_pressure_from_strategy(vapor_pressure_strategy: VaporPressureStrategy | Sequence[VaporPressureStrategy], temperature: float) -> NDArray[np.float64]
```

Compute pure vapor pressure for single or per-species strategies.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L78-L95)
