# particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder

*module*

Builder for the CondensationIsothermalStaggered strategy.

Provides a fluent interface to configure the staggered isothermal
condensation strategy with validation and sensible defaults.

> **Example**
>
> >>> from particula.dynamics.condensation.condensation_builder import (
> ...     CondensationIsothermalStaggeredBuilder,
> ... )
> >>> builder = CondensationIsothermalStaggeredBuilder()
> >>> strategy = (
> ...     builder
> ...     .set_molar_mass(0.018, "kg/mol")
> ...     .set_diffusion_coefficient(2e-5, "m^2/s")
> ...     .set_accommodation_coefficient(1.0)
> ...     .set_theta_mode("random")
> ...     .set_num_batches(10)
> ...     .set_shuffle_each_step(True)
> ...     .set_random_state(42)
> ...     .build()
> ... )

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L1-L1)

## particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L44-L44)

## particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder

*class*

```python
class CondensationIsothermalStaggeredBuilder(BuilderABC, BuilderMolarMassMixin, BuilderDiffusionCoefficientMixin, BuilderAccommodationCoefficientMixin, BuilderUpdateGasesMixin, BuilderNativeCondensationMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderMolarMassMixin`, `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderDiffusionCoefficientMixin`, `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderAccommodationCoefficientMixin`, `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderUpdateGasesMixin`, `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Fluent builder for :class:`CondensationIsothermalStaggered`.

Extends the base condensation builder with staggered-stepping-specific
parameters ``theta_mode``, ``num_batches``, ``shuffle_each_step``, and
``random_state``.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L47-L231)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.num_batches

*attribute* · *instance attribute*

```python
num_batches: int = 1
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L77-L77)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.random_state

*attribute* · *instance attribute*

```python
random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L79-L81)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.shuffle_each_step

*attribute* · *instance attribute*

```python
shuffle_each_step: bool = True
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L78-L78)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.theta_mode

*attribute* · *instance attribute*

```python
theta_mode: str = 'half'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L76-L76)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.molar_mass

*attribute* · *instance attribute*

```python
molar_mass: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.builder_mixin.BuilderMolarMassMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L149-L149)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.diffusion_coefficient

*attribute* · *instance attribute*

```python
diffusion_coefficient: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderDiffusionCoefficientMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L25-L27)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.accommodation_coefficient

*attribute* · *instance attribute*

```python
accommodation_coefficient: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderAccommodationCoefficientMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L57-L59)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.update_gases

*attribute* · *instance attribute*

```python
update_gases: bool = True
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderUpdateGasesMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L84-L84)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder._NATIVE_KEYS

*attribute* · *class attribute* · *instance attribute*

```python
_NATIVE_KEYS = ('species_map', 'activity_strategy', 'surface_strategy', 'vapor_pressure_strategy', 'skip_partitioning_indices')
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L21-L27)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.activity_strategy

*attribute* · *instance attribute*

```python
activity_strategy: ActivityStrategy | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L32-L32)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.skip_partitioning_indices

*attribute* · *instance attribute*

```python
skip_partitioning_indices: tuple[int, ...] | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L37-L37)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.species_map

*attribute* · *instance attribute*

```python
species_map: CondensationSpeciesMap | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L31-L31)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.surface_strategy

*attribute* · *instance attribute*

```python
surface_strategy: SurfaceStrategy | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L33-L33)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.vapor_pressure_strategy

*attribute* · *instance attribute*

```python
vapor_pressure_strategy: VaporPressureStrategy | tuple[VaporPressureStrategy, ...] | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L34-L36)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize the builder with required parameters and defaults.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L62-L81)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.build

*method*

```python
def build() -> CondensationStrategy
```

Validate parameters and create a condensation strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L211-L231)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.set_num_batches

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

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.set_parameters

*method*

```python
def set_parameters(parameters: dict[str, Any]) -> CondensationIsothermalStaggeredBuilder
```

Set required and optional parameters from a dictionary.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_isothermal_staggered_builder.py#L83-L137)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.set_random_state

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

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.set_shuffle_each_step

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

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.set_theta_mode

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

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.check_keys

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

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.pre_build_check

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

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.set_molar_mass

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

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.set_diffusion_coefficient

*method*

```python
def set_diffusion_coefficient(diffusion_coefficient: Union[float, NDArray[np.float64]], diffusion_coefficient_units: str)
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderDiffusionCoefficientMixin`

Set the diffusion coefficient for the condensing species.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L29-L46)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.set_accommodation_coefficient

*method*

```python
def set_accommodation_coefficient(accommodation_coefficient: Union[float, NDArray[np.float64]], accommodation_coefficient_units: Optional[str] = None)
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderAccommodationCoefficientMixin`

Set the dimensionless mass accommodation coefficient.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L61-L73)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.set_update_gases

*method*

```python
def set_update_gases(update_gases: bool, update_gases_units: Optional[str] = None)
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderUpdateGasesMixin`

Set the flag controlling gas-phase updates.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L86-L93)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder._load_native_parameters

*method*

```python
def _load_native_parameters(parameters: dict[str, Any]) -> dict[str, Any]
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Load optional native keys and return remaining builder parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L107-L123)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder._native_strategy_kwargs

*method*

```python
def _native_strategy_kwargs() -> dict[str, Any]
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Return process configuration without constructing data or facades.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L125-L127)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.set_activity_strategy

*method*

```python
def set_activity_strategy(activity_strategy: ActivityStrategy | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Set the activity model that reads the full particle composition.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L52-L63)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.set_skip_partitioning_indices

*method*

```python
def set_skip_partitioning_indices(indices: Sequence[int] | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Detach full-gas skip indices; execution checks lane bounds.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L98-L105)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.set_species_map

*method*

```python
def set_species_map(species_map: CondensationSpeciesMap | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Set immutable full-gas provenance and explicit transfer pairs.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L39-L50)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.set_surface_strategy

*method*

```python
def set_surface_strategy(surface_strategy: SurfaceStrategy | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Set the surface model that reads the full particle composition.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L65-L74)

### particula.dynamics.condensation.condensation_builder.condensation_isothermal_staggered_builder.CondensationIsothermalStaggeredBuilder.set_vapor_pressure_strategy

*method*

```python
def set_vapor_pressure_strategy(vapor_pressure_strategy: VaporPressureStrategy | Sequence[VaporPressureStrategy] | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Set one vector model or a detached ordered model sequence.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L76-L96)
