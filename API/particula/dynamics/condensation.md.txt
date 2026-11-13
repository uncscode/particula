# particula.dynamics.condensation

*module*

Condensation dynamics sub-package.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/__init__.py#L1-L1)

## particula.dynamics.condensation.CondensationFactory

*class*

```python
class CondensationFactory(StrategyFactoryABC[Union[CondensationLatentHeatBuilder, CondensationIsothermalBuilder, CondensationIsothermalStaggeredBuilder], CondensationStrategy])
```

Bases: ``

Re-exported from: `particula.dynamics.condensation.condensation_factories`

Factory class for condensation strategies.

> **Supports strategy types**
>
> - "isothermal": Standard isothermal condensation.
> - "isothermal_staggered": Staggered isothermal condensation with
>   batch stepping for stability.
> - "latent_heat": Condensation with latent-heat coupling.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_factories.py#L15-L53)

### particula.dynamics.condensation.CondensationFactory.get_builders

*method*

```python
def get_builders() -> Dict[str, Union[CondensationLatentHeatBuilder, CondensationIsothermalBuilder, CondensationIsothermalStaggeredBuilder]]
```

Return the mapping of strategy types to builder instances.

**Returns**

- (`Dict[str, Union[CondensationLatentHeatBuilder, CondensationIsothermalBuilder, CondensationIsothermalStaggeredBuilder]]`) — Dictionary mapping condensation strategy names to builders.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_factories.py#L34-L53)

## particula.dynamics.condensation.CondensationIsothermal

*class*

```python
class CondensationIsothermal(CondensationStrategy)
```

Bases: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Re-exported from: `particula.dynamics.condensation.condensation_strategies`

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

### particula.dynamics.condensation.CondensationIsothermal.accommodation_coefficient

*attribute* · *instance attribute*

```python
accommodation_coefficient = accommodation_coefficient
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L230-L230)

### particula.dynamics.condensation.CondensationIsothermal.activity_strategy

*attribute* · *instance attribute*

```python
activity_strategy = activity_strategy
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L233-L233)

### particula.dynamics.condensation.CondensationIsothermal.diffusion_coefficient

*attribute* · *instance attribute*

```python
diffusion_coefficient = diffusion_coefficient
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L229-L229)

### particula.dynamics.condensation.CondensationIsothermal.molar_mass

*attribute* · *instance attribute*

```python
molar_mass = molar_mass
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L228-L228)

### particula.dynamics.condensation.CondensationIsothermal.skip_partitioning_indices

*attribute* · *instance attribute*

```python
skip_partitioning_indices = skip_partitioning_indices
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L232-L232)

### particula.dynamics.condensation.CondensationIsothermal.species_map

*attribute* · *instance attribute*

```python
species_map = species_map
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L236-L236)

### particula.dynamics.condensation.CondensationIsothermal.surface_strategy

*attribute* · *instance attribute*

```python
surface_strategy = surface_strategy
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L234-L234)

### particula.dynamics.condensation.CondensationIsothermal.update_gases

*attribute* · *instance attribute*

```python
update_gases = update_gases
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L231-L231)

### particula.dynamics.condensation.CondensationIsothermal.vapor_pressure_strategy

*attribute* · *instance attribute*

```python
vapor_pressure_strategy = vapor_pressure_strategy
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L235-L235)

### particula.dynamics.condensation.CondensationIsothermal.__init__

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

### particula.dynamics.condensation.CondensationIsothermal.mass_transfer_rate

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

### particula.dynamics.condensation.CondensationIsothermal.rate

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

### particula.dynamics.condensation.CondensationIsothermal.step

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

### particula.dynamics.condensation.CondensationIsothermal._native_latent_heat_values

*method*

```python
def _native_latent_heat_values(gas: GasData, temperature: float) -> NDArray[np.float64] | None
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Validate optional latent heat in the complete gas ordering.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L458-L472)

### particula.dynamics.condensation.CondensationIsothermal._native_mass_transfer_rate

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

### particula.dynamics.condensation.CondensationIsothermal._native_parameters

*method*

```python
def _native_parameters(particle: ParticleData, gas: GasData, temperature: float, pressure: float) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Admit current state and full-gas transport parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L238-L279)

### particula.dynamics.condensation.CondensationIsothermal._native_pressure_terms

*method*

```python
def _native_pressure_terms(particle: ParticleData, gas: GasData, temperature: float, radius: NDArray[np.float64], sources: NDArray[np.intp], targets: NDArray[np.intp]) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Evaluate full composition and mapped pressure lanes in Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L281-L351)

### particula.dynamics.condensation.CondensationIsothermal._native_rate_from_parameters

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

### particula.dynamics.condensation.CondensationIsothermal._native_single_step

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

### particula.dynamics.condensation.CondensationIsothermal._resolve_strategies

*method*

```python
def _resolve_strategies() -> tuple[ActivityStrategy, SurfaceStrategy, VaporPressureStrategy | Sequence[VaporPressureStrategy]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Require explicit process-owned activity, surface and vapor laws.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L539-L561)

### particula.dynamics.condensation.CondensationIsothermal.calculate_pressure_delta

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

### particula.dynamics.condensation.CondensationIsothermal.first_order_mass_transport

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

### particula.dynamics.condensation.CondensationIsothermal.knudsen_number

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

### particula.dynamics.condensation.CondensationIsothermal.mean_free_path

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

## particula.dynamics.condensation.CondensationIsothermalBuilder

*class*

```python
class CondensationIsothermalBuilder(BuilderABC, BuilderMolarMassMixin, BuilderDiffusionCoefficientMixin, BuilderAccommodationCoefficientMixin, BuilderUpdateGasesMixin, BuilderNativeCondensationMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderMolarMassMixin`, `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderDiffusionCoefficientMixin`, `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderAccommodationCoefficientMixin`, `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderUpdateGasesMixin`, `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Re-exported from: `particula.dynamics.condensation.condensation_builder.condensation_isothermal_builder`

Fluent builder for :class:`CondensationIsothermal`.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_builder.py#L23-L78)

### particula.dynamics.condensation.CondensationIsothermalBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.condensation.CondensationIsothermalBuilder.molar_mass

*attribute* · *instance attribute*

```python
molar_mass: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.builder_mixin.BuilderMolarMassMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L149-L149)

### particula.dynamics.condensation.CondensationIsothermalBuilder.diffusion_coefficient

*attribute* · *instance attribute*

```python
diffusion_coefficient: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderDiffusionCoefficientMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L25-L27)

### particula.dynamics.condensation.CondensationIsothermalBuilder.accommodation_coefficient

*attribute* · *instance attribute*

```python
accommodation_coefficient: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderAccommodationCoefficientMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L57-L59)

### particula.dynamics.condensation.CondensationIsothermalBuilder.update_gases

*attribute* · *instance attribute*

```python
update_gases: bool = True
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderUpdateGasesMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L84-L84)

### particula.dynamics.condensation.CondensationIsothermalBuilder._NATIVE_KEYS

*attribute* · *class attribute* · *instance attribute*

```python
_NATIVE_KEYS = ('species_map', 'activity_strategy', 'surface_strategy', 'vapor_pressure_strategy', 'skip_partitioning_indices')
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L21-L27)

### particula.dynamics.condensation.CondensationIsothermalBuilder.activity_strategy

*attribute* · *instance attribute*

```python
activity_strategy: ActivityStrategy | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L32-L32)

### particula.dynamics.condensation.CondensationIsothermalBuilder.skip_partitioning_indices

*attribute* · *instance attribute*

```python
skip_partitioning_indices: tuple[int, ...] | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L37-L37)

### particula.dynamics.condensation.CondensationIsothermalBuilder.species_map

*attribute* · *instance attribute*

```python
species_map: CondensationSpeciesMap | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L31-L31)

### particula.dynamics.condensation.CondensationIsothermalBuilder.surface_strategy

*attribute* · *instance attribute*

```python
surface_strategy: SurfaceStrategy | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L33-L33)

### particula.dynamics.condensation.CondensationIsothermalBuilder.vapor_pressure_strategy

*attribute* · *instance attribute*

```python
vapor_pressure_strategy: VaporPressureStrategy | tuple[VaporPressureStrategy, ...] | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L34-L36)

### particula.dynamics.condensation.CondensationIsothermalBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize the Condensation Isothermal builder.

Sets up the builder with required parameters for creating a
CondensationIsothermal strategy, including molar mass,
diffusion coefficient, and accommodation coefficient.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_builder.py#L33-L50)

### particula.dynamics.condensation.CondensationIsothermalBuilder.build

*method*

```python
def build() -> CondensationStrategy
```

Validate parameters and create a condensation strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_builder.py#L61-L78)

### particula.dynamics.condensation.CondensationIsothermalBuilder.set_parameters

*method*

```python
def set_parameters(parameters: dict[str, Any]) -> CondensationIsothermalBuilder
```

Load transport inputs and optional native process configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_builder.py#L52-L59)

### particula.dynamics.condensation.CondensationIsothermalBuilder.check_keys

*method*

```python
def check_keys(parameters: dict[str, Any]) -> None
```

Inherited from: `particula.abc_builder.BuilderABC`

Check if the keys are present and valid.

**Parameters**

- `parameters` (`dict[str, Any]`) — The parameters dictionary to check.

**Raises**

- `ValueError` — If any required key is missing or if trying to set
an invalid parameter.

**Examples**

```py
builder = Builder()
builder.check_keys({
    "parameter1": 1,
    "parameter2": 2,
})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L53-L94)

### particula.dynamics.condensation.CondensationIsothermalBuilder.pre_build_check

*method*

```python
def pre_build_check() -> None
```

Inherited from: `particula.abc_builder.BuilderABC`

Check if all required attribute parameters are set before building.

**Raises**

- `ValueError` — If any required parameter is missing.

**Examples**

```py
builder = Builder()
builder.pre_build_check()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L132-L151)

### particula.dynamics.condensation.CondensationIsothermalBuilder.set_molar_mass

*method*

```python
def set_molar_mass(molar_mass: Union[float, NDArray[np.float64]], molar_mass_units: str)
```

Inherited from: `particula.builder_mixin.BuilderMolarMassMixin`

Set the molar mass in kilograms per mole.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Positive molar mass value to store.
- `molar_mass_units` (`str`) — Units for ``molar_mass``. Defaults to ``"kg/mol"``.

**Returns**

- `self` (`BuilderMolarMassMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L151-L173)

### particula.dynamics.condensation.CondensationIsothermalBuilder.set_diffusion_coefficient

*method*

```python
def set_diffusion_coefficient(diffusion_coefficient: Union[float, NDArray[np.float64]], diffusion_coefficient_units: str)
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderDiffusionCoefficientMixin`

Set the diffusion coefficient for the condensing species.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L29-L46)

### particula.dynamics.condensation.CondensationIsothermalBuilder.set_accommodation_coefficient

*method*

```python
def set_accommodation_coefficient(accommodation_coefficient: Union[float, NDArray[np.float64]], accommodation_coefficient_units: Optional[str] = None)
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderAccommodationCoefficientMixin`

Set the dimensionless mass accommodation coefficient.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L61-L73)

### particula.dynamics.condensation.CondensationIsothermalBuilder.set_update_gases

*method*

```python
def set_update_gases(update_gases: bool, update_gases_units: Optional[str] = None)
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderUpdateGasesMixin`

Set the flag controlling gas-phase updates.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L86-L93)

### particula.dynamics.condensation.CondensationIsothermalBuilder._load_native_parameters

*method*

```python
def _load_native_parameters(parameters: dict[str, Any]) -> dict[str, Any]
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Load optional native keys and return remaining builder parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L107-L123)

### particula.dynamics.condensation.CondensationIsothermalBuilder._native_strategy_kwargs

*method*

```python
def _native_strategy_kwargs() -> dict[str, Any]
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Return process configuration without constructing data or facades.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L125-L127)

### particula.dynamics.condensation.CondensationIsothermalBuilder.set_activity_strategy

*method*

```python
def set_activity_strategy(activity_strategy: ActivityStrategy | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Set the activity model that reads the full particle composition.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L52-L63)

### particula.dynamics.condensation.CondensationIsothermalBuilder.set_skip_partitioning_indices

*method*

```python
def set_skip_partitioning_indices(indices: Sequence[int] | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Detach full-gas skip indices; execution checks lane bounds.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L98-L105)

### particula.dynamics.condensation.CondensationIsothermalBuilder.set_species_map

*method*

```python
def set_species_map(species_map: CondensationSpeciesMap | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Set immutable full-gas provenance and explicit transfer pairs.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L39-L50)

### particula.dynamics.condensation.CondensationIsothermalBuilder.set_surface_strategy

*method*

```python
def set_surface_strategy(surface_strategy: SurfaceStrategy | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Set the surface model that reads the full particle composition.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L65-L74)

### particula.dynamics.condensation.CondensationIsothermalBuilder.set_vapor_pressure_strategy

*method*

```python
def set_vapor_pressure_strategy(vapor_pressure_strategy: VaporPressureStrategy | Sequence[VaporPressureStrategy] | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Set one vector model or a detached ordered model sequence.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L76-L96)

## particula.dynamics.condensation.CondensationIsothermalStaggered

*class*

```python
class CondensationIsothermalStaggered(CondensationStrategy)
```

Bases: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Re-exported from: `particula.dynamics.condensation.condensation_strategies`

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

### particula.dynamics.condensation.CondensationIsothermalStaggered.VALID_THETA_MODES

*attribute* · *class attribute* · *instance attribute*

```python
VALID_THETA_MODES = ('half', 'random', 'batch')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1106-L1106)

### particula.dynamics.condensation.CondensationIsothermalStaggered.num_batches

*attribute* · *instance attribute*

```python
num_batches = num_batches
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1182-L1182)

### particula.dynamics.condensation.CondensationIsothermalStaggered.random_state

*attribute* · *instance attribute*

```python
random_state = random_state
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1184-L1184)

### particula.dynamics.condensation.CondensationIsothermalStaggered.shuffle_each_step

*attribute* · *instance attribute*

```python
shuffle_each_step = shuffle_each_step
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1183-L1183)

### particula.dynamics.condensation.CondensationIsothermalStaggered.theta_mode

*attribute* · *instance attribute*

```python
theta_mode = theta_mode
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1181-L1181)

### particula.dynamics.condensation.CondensationIsothermalStaggered.accommodation_coefficient

*attribute* · *instance attribute*

```python
accommodation_coefficient = accommodation_coefficient
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L230-L230)

### particula.dynamics.condensation.CondensationIsothermalStaggered.activity_strategy

*attribute* · *instance attribute*

```python
activity_strategy = activity_strategy
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L233-L233)

### particula.dynamics.condensation.CondensationIsothermalStaggered.diffusion_coefficient

*attribute* · *instance attribute*

```python
diffusion_coefficient = diffusion_coefficient
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L229-L229)

### particula.dynamics.condensation.CondensationIsothermalStaggered.molar_mass

*attribute* · *instance attribute*

```python
molar_mass = molar_mass
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L228-L228)

### particula.dynamics.condensation.CondensationIsothermalStaggered.skip_partitioning_indices

*attribute* · *instance attribute*

```python
skip_partitioning_indices = skip_partitioning_indices
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L232-L232)

### particula.dynamics.condensation.CondensationIsothermalStaggered.species_map

*attribute* · *instance attribute*

```python
species_map = species_map
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L236-L236)

### particula.dynamics.condensation.CondensationIsothermalStaggered.surface_strategy

*attribute* · *instance attribute*

```python
surface_strategy = surface_strategy
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L234-L234)

### particula.dynamics.condensation.CondensationIsothermalStaggered.update_gases

*attribute* · *instance attribute*

```python
update_gases = update_gases
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L231-L231)

### particula.dynamics.condensation.CondensationIsothermalStaggered.vapor_pressure_strategy

*attribute* · *instance attribute*

```python
vapor_pressure_strategy = vapor_pressure_strategy
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L235-L235)

### particula.dynamics.condensation.CondensationIsothermalStaggered.__init__

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

### particula.dynamics.condensation.CondensationIsothermalStaggered._get_theta_values

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

### particula.dynamics.condensation.CondensationIsothermalStaggered._make_batches

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

### particula.dynamics.condensation.CondensationIsothermalStaggered._native_staggered_step

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

### particula.dynamics.condensation.CondensationIsothermalStaggered._validate_native_staggered_controls

*method*

```python
def _validate_native_staggered_controls() -> None
```

Revalidate mutable controls before drawing randomness or writing.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1296-L1320)

### particula.dynamics.condensation.CondensationIsothermalStaggered._validate_num_batches

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

### particula.dynamics.condensation.CondensationIsothermalStaggered.mass_transfer_rate

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

### particula.dynamics.condensation.CondensationIsothermalStaggered.rate

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

### particula.dynamics.condensation.CondensationIsothermalStaggered.step

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

### particula.dynamics.condensation.CondensationIsothermalStaggered._native_latent_heat_values

*method*

```python
def _native_latent_heat_values(gas: GasData, temperature: float) -> NDArray[np.float64] | None
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Validate optional latent heat in the complete gas ordering.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L458-L472)

### particula.dynamics.condensation.CondensationIsothermalStaggered._native_mass_transfer_rate

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

### particula.dynamics.condensation.CondensationIsothermalStaggered._native_parameters

*method*

```python
def _native_parameters(particle: ParticleData, gas: GasData, temperature: float, pressure: float) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Admit current state and full-gas transport parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L238-L279)

### particula.dynamics.condensation.CondensationIsothermalStaggered._native_pressure_terms

*method*

```python
def _native_pressure_terms(particle: ParticleData, gas: GasData, temperature: float, radius: NDArray[np.float64], sources: NDArray[np.intp], targets: NDArray[np.intp]) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Evaluate full composition and mapped pressure lanes in Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L281-L351)

### particula.dynamics.condensation.CondensationIsothermalStaggered._native_rate_from_parameters

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

### particula.dynamics.condensation.CondensationIsothermalStaggered._native_single_step

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

### particula.dynamics.condensation.CondensationIsothermalStaggered._resolve_strategies

*method*

```python
def _resolve_strategies() -> tuple[ActivityStrategy, SurfaceStrategy, VaporPressureStrategy | Sequence[VaporPressureStrategy]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Require explicit process-owned activity, surface and vapor laws.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L539-L561)

### particula.dynamics.condensation.CondensationIsothermalStaggered.calculate_pressure_delta

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

### particula.dynamics.condensation.CondensationIsothermalStaggered.first_order_mass_transport

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

### particula.dynamics.condensation.CondensationIsothermalStaggered.knudsen_number

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

### particula.dynamics.condensation.CondensationIsothermalStaggered.mean_free_path

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

## particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder

*class*

```python
class CondensationIsothermalStaggeredBuilder(BuilderABC, BuilderMolarMassMixin, BuilderDiffusionCoefficientMixin, BuilderAccommodationCoefficientMixin, BuilderUpdateGasesMixin, BuilderNativeCondensationMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderMolarMassMixin`, `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderDiffusionCoefficientMixin`, `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderAccommodationCoefficientMixin`, `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderUpdateGasesMixin`, `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Re-exported from: `particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder`

Fluent builder for :class:`CondensationIsothermalStaggered`.

Extends the base condensation builder with staggered-stepping-specific
parameters ``theta_mode``, ``num_batches``, ``shuffle_each_step``, and
``random_state``.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L47-L231)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.num_batches

*attribute* · *instance attribute*

```python
num_batches: int = 1
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L77-L77)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.random_state

*attribute* · *instance attribute*

```python
random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L79-L81)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.shuffle_each_step

*attribute* · *instance attribute*

```python
shuffle_each_step: bool = True
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L78-L78)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.theta_mode

*attribute* · *instance attribute*

```python
theta_mode: str = 'half'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L76-L76)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.molar_mass

*attribute* · *instance attribute*

```python
molar_mass: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.builder_mixin.BuilderMolarMassMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L149-L149)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.diffusion_coefficient

*attribute* · *instance attribute*

```python
diffusion_coefficient: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderDiffusionCoefficientMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L25-L27)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.accommodation_coefficient

*attribute* · *instance attribute*

```python
accommodation_coefficient: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderAccommodationCoefficientMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L57-L59)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.update_gases

*attribute* · *instance attribute*

```python
update_gases: bool = True
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderUpdateGasesMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L84-L84)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder._NATIVE_KEYS

*attribute* · *class attribute* · *instance attribute*

```python
_NATIVE_KEYS = ('species_map', 'activity_strategy', 'surface_strategy', 'vapor_pressure_strategy', 'skip_partitioning_indices')
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L21-L27)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.activity_strategy

*attribute* · *instance attribute*

```python
activity_strategy: ActivityStrategy | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L32-L32)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.skip_partitioning_indices

*attribute* · *instance attribute*

```python
skip_partitioning_indices: tuple[int, ...] | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L37-L37)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.species_map

*attribute* · *instance attribute*

```python
species_map: CondensationSpeciesMap | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L31-L31)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.surface_strategy

*attribute* · *instance attribute*

```python
surface_strategy: SurfaceStrategy | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L33-L33)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.vapor_pressure_strategy

*attribute* · *instance attribute*

```python
vapor_pressure_strategy: VaporPressureStrategy | tuple[VaporPressureStrategy, ...] | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L34-L36)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize the builder with required parameters and defaults.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L62-L81)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.build

*method*

```python
def build() -> CondensationStrategy
```

Validate parameters and create a condensation strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L211-L231)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.set_num_batches

*method*

```python
def set_num_batches(num_batches: int) -> CondensationIsothermalStaggeredBuilder
```

Set the number of batches for staggered updates.

**Parameters**

- `num_batches` (`int`) — Number of batches; must be at least 1.

**Returns**

- (`CondensationIsothermalStaggeredBuilder`) — The builder instance for chaining.

**Raises**

- `ValueError` — If ``num_batches`` is less than 1.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L161-L178)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.set_parameters

*method*

```python
def set_parameters(parameters: dict[str, Any]) -> CondensationIsothermalStaggeredBuilder
```

Set required and optional parameters from a dictionary.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L83-L137)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.set_random_state

*method*

```python
def set_random_state(random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]]) -> CondensationIsothermalStaggeredBuilder
```

Set the random state for reproducibility.

**Parameters**

- `random_state` (`Optional[Union[int, np.random.Generator, np.random.RandomState]]`) — Seed or RNG controlling random theta generation.

**Returns**

- (`CondensationIsothermalStaggeredBuilder`) — The builder instance for chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L194-L209)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.set_shuffle_each_step

*method*

```python
def set_shuffle_each_step(shuffle: bool) -> CondensationIsothermalStaggeredBuilder
```

Enable or disable shuffling at each step.

**Parameters**

- `shuffle` (`bool`) — Whether to shuffle particle order every step.

**Returns**

- (`CondensationIsothermalStaggeredBuilder`) — The builder instance for chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L180-L192)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.set_theta_mode

*method*

```python
def set_theta_mode(theta_mode: str) -> CondensationIsothermalStaggeredBuilder
```

Set the staggered stepping mode.

**Parameters**

- `theta_mode` (`str`) — One of ``("half", "random", "batch")``.

**Returns**

- (`CondensationIsothermalStaggeredBuilder`) — The builder instance for chaining.

**Raises**

- `ValueError` — If ``theta_mode`` is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L139-L159)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.check_keys

*method*

```python
def check_keys(parameters: dict[str, Any]) -> None
```

Inherited from: `particula.abc_builder.BuilderABC`

Check if the keys are present and valid.

**Parameters**

- `parameters` (`dict[str, Any]`) — The parameters dictionary to check.

**Raises**

- `ValueError` — If any required key is missing or if trying to set
an invalid parameter.

**Examples**

```py
builder = Builder()
builder.check_keys({
    "parameter1": 1,
    "parameter2": 2,
})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L53-L94)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.pre_build_check

*method*

```python
def pre_build_check() -> None
```

Inherited from: `particula.abc_builder.BuilderABC`

Check if all required attribute parameters are set before building.

**Raises**

- `ValueError` — If any required parameter is missing.

**Examples**

```py
builder = Builder()
builder.pre_build_check()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L132-L151)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.set_molar_mass

*method*

```python
def set_molar_mass(molar_mass: Union[float, NDArray[np.float64]], molar_mass_units: str)
```

Inherited from: `particula.builder_mixin.BuilderMolarMassMixin`

Set the molar mass in kilograms per mole.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Positive molar mass value to store.
- `molar_mass_units` (`str`) — Units for ``molar_mass``. Defaults to ``"kg/mol"``.

**Returns**

- `self` (`BuilderMolarMassMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L151-L173)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.set_diffusion_coefficient

*method*

```python
def set_diffusion_coefficient(diffusion_coefficient: Union[float, NDArray[np.float64]], diffusion_coefficient_units: str)
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderDiffusionCoefficientMixin`

Set the diffusion coefficient for the condensing species.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L29-L46)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.set_accommodation_coefficient

*method*

```python
def set_accommodation_coefficient(accommodation_coefficient: Union[float, NDArray[np.float64]], accommodation_coefficient_units: Optional[str] = None)
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderAccommodationCoefficientMixin`

Set the dimensionless mass accommodation coefficient.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L61-L73)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.set_update_gases

*method*

```python
def set_update_gases(update_gases: bool, update_gases_units: Optional[str] = None)
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderUpdateGasesMixin`

Set the flag controlling gas-phase updates.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L86-L93)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder._load_native_parameters

*method*

```python
def _load_native_parameters(parameters: dict[str, Any]) -> dict[str, Any]
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Load optional native keys and return remaining builder parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L107-L123)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder._native_strategy_kwargs

*method*

```python
def _native_strategy_kwargs() -> dict[str, Any]
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Return process configuration without constructing data or facades.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L125-L127)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.set_activity_strategy

*method*

```python
def set_activity_strategy(activity_strategy: ActivityStrategy | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Set the activity model that reads the full particle composition.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L52-L63)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.set_skip_partitioning_indices

*method*

```python
def set_skip_partitioning_indices(indices: Sequence[int] | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Detach full-gas skip indices; execution checks lane bounds.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L98-L105)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.set_species_map

*method*

```python
def set_species_map(species_map: CondensationSpeciesMap | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Set immutable full-gas provenance and explicit transfer pairs.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L39-L50)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.set_surface_strategy

*method*

```python
def set_surface_strategy(surface_strategy: SurfaceStrategy | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Set the surface model that reads the full particle composition.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L65-L74)

### particula.dynamics.condensation.CondensationIsothermalStaggeredBuilder.set_vapor_pressure_strategy

*method*

```python
def set_vapor_pressure_strategy(vapor_pressure_strategy: VaporPressureStrategy | Sequence[VaporPressureStrategy] | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Set one vector model or a detached ordered model sequence.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L76-L96)

## particula.dynamics.condensation.CondensationLatentHeat

*class*

```python
class CondensationLatentHeat(CondensationStrategy)
```

Bases: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Re-exported from: `particula.dynamics.condensation.condensation_strategies`

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

### particula.dynamics.condensation.CondensationLatentHeat._latent_heat_strategy

*attribute* · *instance attribute*

```python
_latent_heat_strategy = self._resolve_latent_heat_strategy(latent_heat_strategy=latent_heat_strategy, latent_heat=latent_heat)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1595-L1598)

### particula.dynamics.condensation.CondensationLatentHeat.last_latent_heat_energy

*attribute* · *instance attribute*

```python
last_latent_heat_energy = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1599-L1599)

### particula.dynamics.condensation.CondensationLatentHeat.latent_heat_input

*attribute* · *instance attribute*

```python
latent_heat_input = latent_heat
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1594-L1594)

### particula.dynamics.condensation.CondensationLatentHeat.latent_heat_strategy_input

*attribute* · *instance attribute*

```python
latent_heat_strategy_input = latent_heat_strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L1593-L1593)

### particula.dynamics.condensation.CondensationLatentHeat.accommodation_coefficient

*attribute* · *instance attribute*

```python
accommodation_coefficient = accommodation_coefficient
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L230-L230)

### particula.dynamics.condensation.CondensationLatentHeat.activity_strategy

*attribute* · *instance attribute*

```python
activity_strategy = activity_strategy
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L233-L233)

### particula.dynamics.condensation.CondensationLatentHeat.diffusion_coefficient

*attribute* · *instance attribute*

```python
diffusion_coefficient = diffusion_coefficient
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L229-L229)

### particula.dynamics.condensation.CondensationLatentHeat.molar_mass

*attribute* · *instance attribute*

```python
molar_mass = molar_mass
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L228-L228)

### particula.dynamics.condensation.CondensationLatentHeat.skip_partitioning_indices

*attribute* · *instance attribute*

```python
skip_partitioning_indices = skip_partitioning_indices
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L232-L232)

### particula.dynamics.condensation.CondensationLatentHeat.species_map

*attribute* · *instance attribute*

```python
species_map = species_map
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L236-L236)

### particula.dynamics.condensation.CondensationLatentHeat.surface_strategy

*attribute* · *instance attribute*

```python
surface_strategy = surface_strategy
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L234-L234)

### particula.dynamics.condensation.CondensationLatentHeat.update_gases

*attribute* · *instance attribute*

```python
update_gases = update_gases
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L231-L231)

### particula.dynamics.condensation.CondensationLatentHeat.vapor_pressure_strategy

*attribute* · *instance attribute*

```python
vapor_pressure_strategy = vapor_pressure_strategy
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L235-L235)

### particula.dynamics.condensation.CondensationLatentHeat.__init__

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

### particula.dynamics.condensation.CondensationLatentHeat._resolve_latent_heat_strategy

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

### particula.dynamics.condensation.CondensationLatentHeat.mass_transfer_rate

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

### particula.dynamics.condensation.CondensationLatentHeat.rate

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

### particula.dynamics.condensation.CondensationLatentHeat.step

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

### particula.dynamics.condensation.CondensationLatentHeat._native_latent_heat_values

*method*

```python
def _native_latent_heat_values(gas: GasData, temperature: float) -> NDArray[np.float64] | None
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Validate optional latent heat in the complete gas ordering.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L458-L472)

### particula.dynamics.condensation.CondensationLatentHeat._native_mass_transfer_rate

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

### particula.dynamics.condensation.CondensationLatentHeat._native_parameters

*method*

```python
def _native_parameters(particle: ParticleData, gas: GasData, temperature: float, pressure: float) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Admit current state and full-gas transport parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L238-L279)

### particula.dynamics.condensation.CondensationLatentHeat._native_pressure_terms

*method*

```python
def _native_pressure_terms(particle: ParticleData, gas: GasData, temperature: float, radius: NDArray[np.float64], sources: NDArray[np.intp], targets: NDArray[np.intp]) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Evaluate full composition and mapped pressure lanes in Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L281-L351)

### particula.dynamics.condensation.CondensationLatentHeat._native_rate_from_parameters

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

### particula.dynamics.condensation.CondensationLatentHeat._native_single_step

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

### particula.dynamics.condensation.CondensationLatentHeat._resolve_strategies

*method*

```python
def _resolve_strategies() -> tuple[ActivityStrategy, SurfaceStrategy, VaporPressureStrategy | Sequence[VaporPressureStrategy]]
```

Inherited from: `particula.dynamics.condensation.condensation_strategies.CondensationStrategy`

Require explicit process-owned activity, surface and vapor laws.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L539-L561)

### particula.dynamics.condensation.CondensationLatentHeat.calculate_pressure_delta

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

### particula.dynamics.condensation.CondensationLatentHeat.first_order_mass_transport

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

### particula.dynamics.condensation.CondensationLatentHeat.knudsen_number

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

### particula.dynamics.condensation.CondensationLatentHeat.mean_free_path

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

## particula.dynamics.condensation.CondensationLatentHeatBuilder

*class*

```python
class CondensationLatentHeatBuilder(BuilderABC, BuilderMolarMassMixin, BuilderDiffusionCoefficientMixin, BuilderAccommodationCoefficientMixin, BuilderUpdateGasesMixin, BuilderNativeCondensationMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderMolarMassMixin`, `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderDiffusionCoefficientMixin`, `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderAccommodationCoefficientMixin`, `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderUpdateGasesMixin`, `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Re-exported from: `particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder`

Fluent builder for :class:`CondensationLatentHeat`.

**Attributes**

- `latent_heat_strategy` (`LatentHeatStrategy | None`) — Optional latent heat strategy passed directly to the constructed condensation strategy.
- `latent_heat` (`float | None`) — Optional positive scalar latent heat fallback in J/kg.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py#L31-L248)

### particula.dynamics.condensation.CondensationLatentHeatBuilder._latent_heat_explicitly_set

*attribute* · *instance attribute*

```python
_latent_heat_explicitly_set = False
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py#L68-L68)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.latent_heat

*attribute* · *instance attribute*

```python
latent_heat: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py#L67-L67)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.latent_heat_strategy

*attribute* · *instance attribute*

```python
latent_heat_strategy: LatentHeatStrategy | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py#L66-L66)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.molar_mass

*attribute* · *instance attribute*

```python
molar_mass: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.builder_mixin.BuilderMolarMassMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L149-L149)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.diffusion_coefficient

*attribute* · *instance attribute*

```python
diffusion_coefficient: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderDiffusionCoefficientMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L25-L27)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.accommodation_coefficient

*attribute* · *instance attribute*

```python
accommodation_coefficient: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderAccommodationCoefficientMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L57-L59)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.update_gases

*attribute* · *instance attribute*

```python
update_gases: bool = True
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderUpdateGasesMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L84-L84)

### particula.dynamics.condensation.CondensationLatentHeatBuilder._NATIVE_KEYS

*attribute* · *class attribute* · *instance attribute*

```python
_NATIVE_KEYS = ('species_map', 'activity_strategy', 'surface_strategy', 'vapor_pressure_strategy', 'skip_partitioning_indices')
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L21-L27)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.activity_strategy

*attribute* · *instance attribute*

```python
activity_strategy: ActivityStrategy | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L32-L32)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.skip_partitioning_indices

*attribute* · *instance attribute*

```python
skip_partitioning_indices: tuple[int, ...] | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L37-L37)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.species_map

*attribute* · *instance attribute*

```python
species_map: CondensationSpeciesMap | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L31-L31)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.surface_strategy

*attribute* · *instance attribute*

```python
surface_strategy: SurfaceStrategy | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L33-L33)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.vapor_pressure_strategy

*attribute* · *instance attribute*

```python
vapor_pressure_strategy: VaporPressureStrategy | tuple[VaporPressureStrategy, ...] | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L34-L36)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize the builder with required shared parameters.

The builder starts with the common condensation transport inputs and
no latent heat override so :meth:`build` only forwards optional latent
heat configuration when it was explicitly provided.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py#L47-L68)

### particula.dynamics.condensation.CondensationLatentHeatBuilder._reset_optional_latent_heat_state

*method*

```python
def _reset_optional_latent_heat_state() -> None
```

Clear optional latent-heat settings for a fresh parameter load.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py#L70-L75)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.build

*method*

```python
def build() -> CondensationLatentHeat
```

Validate parameters and create a latent-heat strategy.

**Returns**

- (`CondensationLatentHeat`) — A configured :class:`CondensationLatentHeat` instance.

> **Notes**
>
> An explicit ``latent_heat_strategy`` is forwarded unchanged. A
> scalar ``latent_heat`` is only passed when
> :meth:`set_latent_heat` was called successfully.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py#L218-L248)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.set_latent_heat

*method*

```python
def set_latent_heat(latent_heat: float | NDArray[np.float64] | None) -> CondensationLatentHeatBuilder
```

Set a positive finite scalar latent heat fallback.

**Parameters**

- `latent_heat` (`float | NDArray[np.float64] | None`) — Scalar latent heat value in J/kg.

**Returns**

- (`CondensationLatentHeatBuilder`) — The builder instance for chaining.

**Raises**

- `ValueError` — If ``latent_heat`` is None, array-like, non-finite,
zero, or negative.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py#L175-L216)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.set_latent_heat_strategy

*method*

```python
def set_latent_heat_strategy(latent_heat_strategy: LatentHeatStrategy | None) -> CondensationLatentHeatBuilder
```

Store a latent heat strategy for direct passthrough.

**Parameters**

- `latent_heat_strategy` (`LatentHeatStrategy | None`) — Strategy object used by :class:`CondensationLatentHeat`.

**Returns**

- (`CondensationLatentHeatBuilder`) — The builder instance for chaining.

> **Notes**
>
> Passing ``None`` clears any previously stored strategy.

**Raises**

- `TypeError` — If ``latent_heat_strategy`` is not ``None`` and does not
implement :class:`LatentHeatStrategy`.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py#L146-L173)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.set_parameters

*method*

```python
def set_parameters(parameters: dict[str, Any]) -> CondensationLatentHeatBuilder
```

Set required and optional parameters from a dictionary.

**Parameters**

- `parameters` (`dict[str, Any]`) — Required shared builder parameters plus optional ``update_gases``, ``latent_heat_strategy``, and ``latent_heat``.

**Returns**

- (`CondensationLatentHeatBuilder`) — The builder instance for chaining.

**Raises**

- `ValueError` — If required keys are missing or any key is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py#L77-L144)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.check_keys

*method*

```python
def check_keys(parameters: dict[str, Any]) -> None
```

Inherited from: `particula.abc_builder.BuilderABC`

Check if the keys are present and valid.

**Parameters**

- `parameters` (`dict[str, Any]`) — The parameters dictionary to check.

**Raises**

- `ValueError` — If any required key is missing or if trying to set
an invalid parameter.

**Examples**

```py
builder = Builder()
builder.check_keys({
    "parameter1": 1,
    "parameter2": 2,
})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L53-L94)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.pre_build_check

*method*

```python
def pre_build_check() -> None
```

Inherited from: `particula.abc_builder.BuilderABC`

Check if all required attribute parameters are set before building.

**Raises**

- `ValueError` — If any required parameter is missing.

**Examples**

```py
builder = Builder()
builder.pre_build_check()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L132-L151)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.set_molar_mass

*method*

```python
def set_molar_mass(molar_mass: Union[float, NDArray[np.float64]], molar_mass_units: str)
```

Inherited from: `particula.builder_mixin.BuilderMolarMassMixin`

Set the molar mass in kilograms per mole.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Positive molar mass value to store.
- `molar_mass_units` (`str`) — Units for ``molar_mass``. Defaults to ``"kg/mol"``.

**Returns**

- `self` (`BuilderMolarMassMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L151-L173)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.set_diffusion_coefficient

*method*

```python
def set_diffusion_coefficient(diffusion_coefficient: Union[float, NDArray[np.float64]], diffusion_coefficient_units: str)
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderDiffusionCoefficientMixin`

Set the diffusion coefficient for the condensing species.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L29-L46)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.set_accommodation_coefficient

*method*

```python
def set_accommodation_coefficient(accommodation_coefficient: Union[float, NDArray[np.float64]], accommodation_coefficient_units: Optional[str] = None)
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderAccommodationCoefficientMixin`

Set the dimensionless mass accommodation coefficient.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L61-L73)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.set_update_gases

*method*

```python
def set_update_gases(update_gases: bool, update_gases_units: Optional[str] = None)
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderUpdateGasesMixin`

Set the flag controlling gas-phase updates.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L86-L93)

### particula.dynamics.condensation.CondensationLatentHeatBuilder._load_native_parameters

*method*

```python
def _load_native_parameters(parameters: dict[str, Any]) -> dict[str, Any]
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Load optional native keys and return remaining builder parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L107-L123)

### particula.dynamics.condensation.CondensationLatentHeatBuilder._native_strategy_kwargs

*method*

```python
def _native_strategy_kwargs() -> dict[str, Any]
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Return process configuration without constructing data or facades.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L125-L127)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.set_activity_strategy

*method*

```python
def set_activity_strategy(activity_strategy: ActivityStrategy | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Set the activity model that reads the full particle composition.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L52-L63)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.set_skip_partitioning_indices

*method*

```python
def set_skip_partitioning_indices(indices: Sequence[int] | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Detach full-gas skip indices; execution checks lane bounds.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L98-L105)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.set_species_map

*method*

```python
def set_species_map(species_map: CondensationSpeciesMap | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Set immutable full-gas provenance and explicit transfer pairs.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L39-L50)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.set_surface_strategy

*method*

```python
def set_surface_strategy(surface_strategy: SurfaceStrategy | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Set the surface model that reads the full particle composition.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L65-L74)

### particula.dynamics.condensation.CondensationLatentHeatBuilder.set_vapor_pressure_strategy

*method*

```python
def set_vapor_pressure_strategy(vapor_pressure_strategy: VaporPressureStrategy | Sequence[VaporPressureStrategy] | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Set one vector model or a detached ordered model sequence.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L76-L96)

## particula.dynamics.condensation.CondensationStrategy

*class*

```python
class CondensationStrategy(ABC)
```

Bases: `ABC`

Re-exported from: `particula.dynamics.condensation.condensation_strategies`

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

### particula.dynamics.condensation.CondensationStrategy.accommodation_coefficient

*attribute* · *instance attribute*

```python
accommodation_coefficient = accommodation_coefficient
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L230-L230)

### particula.dynamics.condensation.CondensationStrategy.activity_strategy

*attribute* · *instance attribute*

```python
activity_strategy = activity_strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L233-L233)

### particula.dynamics.condensation.CondensationStrategy.diffusion_coefficient

*attribute* · *instance attribute*

```python
diffusion_coefficient = diffusion_coefficient
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L229-L229)

### particula.dynamics.condensation.CondensationStrategy.molar_mass

*attribute* · *instance attribute*

```python
molar_mass = molar_mass
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L228-L228)

### particula.dynamics.condensation.CondensationStrategy.skip_partitioning_indices

*attribute* · *instance attribute*

```python
skip_partitioning_indices = skip_partitioning_indices
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L232-L232)

### particula.dynamics.condensation.CondensationStrategy.species_map

*attribute* · *instance attribute*

```python
species_map = species_map
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L236-L236)

### particula.dynamics.condensation.CondensationStrategy.surface_strategy

*attribute* · *instance attribute*

```python
surface_strategy = surface_strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L234-L234)

### particula.dynamics.condensation.CondensationStrategy.update_gases

*attribute* · *instance attribute*

```python
update_gases = update_gases
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L231-L231)

### particula.dynamics.condensation.CondensationStrategy.vapor_pressure_strategy

*attribute* · *instance attribute*

```python
vapor_pressure_strategy = vapor_pressure_strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L235-L235)

### particula.dynamics.condensation.CondensationStrategy.__init__

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

### particula.dynamics.condensation.CondensationStrategy._native_latent_heat_values

*method*

```python
def _native_latent_heat_values(gas: GasData, temperature: float) -> NDArray[np.float64] | None
```

Validate optional latent heat in the complete gas ordering.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L458-L472)

### particula.dynamics.condensation.CondensationStrategy._native_mass_transfer_rate

*method*

```python
def _native_mass_transfer_rate(particle: ParticleData, gas_species: GasData, temperature: float, pressure: float, dynamic_viscosity: float | None = None) -> NDArray[np.float64]
```

Return native per-particle kg/s in full particle species order.

Transport and vapor parameters follow the full gas ordering. Activity
and surface strategies see the full particle composition. Pair-specific
scalar transport calls avoid accidental particle/species broadcasting.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L353-L390)

### particula.dynamics.condensation.CondensationStrategy._native_parameters

*method*

```python
def _native_parameters(particle: ParticleData, gas: GasData, temperature: float, pressure: float) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
```

Admit current state and full-gas transport parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L238-L279)

### particula.dynamics.condensation.CondensationStrategy._native_pressure_terms

*method*

```python
def _native_pressure_terms(particle: ParticleData, gas: GasData, temperature: float, radius: NDArray[np.float64], sources: NDArray[np.intp], targets: NDArray[np.intp]) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Evaluate full composition and mapped pressure lanes in Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L281-L351)

### particula.dynamics.condensation.CondensationStrategy._native_rate_from_parameters

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

### particula.dynamics.condensation.CondensationStrategy._native_single_step

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

### particula.dynamics.condensation.CondensationStrategy._resolve_strategies

*method*

```python
def _resolve_strategies() -> tuple[ActivityStrategy, SurfaceStrategy, VaporPressureStrategy | Sequence[VaporPressureStrategy]]
```

Require explicit process-owned activity, surface and vapor laws.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_strategies.py#L539-L561)

### particula.dynamics.condensation.CondensationStrategy.calculate_pressure_delta

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

### particula.dynamics.condensation.CondensationStrategy.first_order_mass_transport

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

### particula.dynamics.condensation.CondensationStrategy.knudsen_number

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

### particula.dynamics.condensation.CondensationStrategy.mass_transfer_rate

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

### particula.dynamics.condensation.CondensationStrategy.mean_free_path

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

### particula.dynamics.condensation.CondensationStrategy.rate

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

### particula.dynamics.condensation.CondensationStrategy.step

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

**Modules**

- `particula.dynamics.condensation._native_step`
- `particula.dynamics.condensation.condensation_builder`
- `particula.dynamics.condensation.condensation_factories`
- `particula.dynamics.condensation.condensation_strategies`
- `particula.dynamics.condensation.mass_transfer`
- `particula.dynamics.condensation.mass_transfer_utils`
- `particula.dynamics.condensation.species_mapping`
