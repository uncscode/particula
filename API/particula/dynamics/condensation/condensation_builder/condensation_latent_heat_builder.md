# particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder

*module*

Builder for latent-heat-aware condensation strategies.

This builder configures shared condensation transport properties together
with either a latent heat strategy instance or a validated scalar latent
heat fallback before constructing :class:`CondensationLatentHeat`.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py#L1-L1)

## particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py#L28-L28)

## particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder

*class*

```python
class CondensationLatentHeatBuilder(BuilderABC, BuilderMolarMassMixin, BuilderDiffusionCoefficientMixin, BuilderAccommodationCoefficientMixin, BuilderUpdateGasesMixin, BuilderNativeCondensationMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderMolarMassMixin`, `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderDiffusionCoefficientMixin`, `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderAccommodationCoefficientMixin`, `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderUpdateGasesMixin`, `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Fluent builder for :class:`CondensationLatentHeat`.

**Attributes**

- `latent_heat_strategy` (`LatentHeatStrategy | None`) — Optional latent heat strategy passed directly to the constructed condensation strategy.
- `latent_heat` (`float | None`) — Optional positive scalar latent heat fallback in J/kg.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py#L31-L248)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder._latent_heat_explicitly_set

*attribute* · *instance attribute*

```python
_latent_heat_explicitly_set = False
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py#L68-L68)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.latent_heat

*attribute* · *instance attribute*

```python
latent_heat: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py#L67-L67)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.latent_heat_strategy

*attribute* · *instance attribute*

```python
latent_heat_strategy: LatentHeatStrategy | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py#L66-L66)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.molar_mass

*attribute* · *instance attribute*

```python
molar_mass: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.builder_mixin.BuilderMolarMassMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L149-L149)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.diffusion_coefficient

*attribute* · *instance attribute*

```python
diffusion_coefficient: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderDiffusionCoefficientMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L25-L27)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.accommodation_coefficient

*attribute* · *instance attribute*

```python
accommodation_coefficient: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderAccommodationCoefficientMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L57-L59)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.update_gases

*attribute* · *instance attribute*

```python
update_gases: bool = True
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderUpdateGasesMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L84-L84)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder._NATIVE_KEYS

*attribute* · *class attribute* · *instance attribute*

```python
_NATIVE_KEYS = ('species_map', 'activity_strategy', 'surface_strategy', 'vapor_pressure_strategy', 'skip_partitioning_indices')
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L21-L27)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.activity_strategy

*attribute* · *instance attribute*

```python
activity_strategy: ActivityStrategy | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L32-L32)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.skip_partitioning_indices

*attribute* · *instance attribute*

```python
skip_partitioning_indices: tuple[int, ...] | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L37-L37)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.species_map

*attribute* · *instance attribute*

```python
species_map: CondensationSpeciesMap | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L31-L31)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.surface_strategy

*attribute* · *instance attribute*

```python
surface_strategy: SurfaceStrategy | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L33-L33)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.vapor_pressure_strategy

*attribute* · *instance attribute*

```python
vapor_pressure_strategy: VaporPressureStrategy | tuple[VaporPressureStrategy, ...] | None = None
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L34-L36)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize the builder with required shared parameters.

The builder starts with the common condensation transport inputs and
no latent heat override so :meth:`build` only forwards optional latent
heat configuration when it was explicitly provided.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py#L47-L68)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder._reset_optional_latent_heat_state

*method*

```python
def _reset_optional_latent_heat_state() -> None
```

Clear optional latent-heat settings for a fresh parameter load.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py#L70-L75)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.build

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

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.set_latent_heat

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

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.set_latent_heat_strategy

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

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.set_parameters

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

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.check_keys

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

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.pre_build_check

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

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.set_molar_mass

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

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.set_diffusion_coefficient

*method*

```python
def set_diffusion_coefficient(diffusion_coefficient: Union[float, NDArray[np.float64]], diffusion_coefficient_units: str)
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderDiffusionCoefficientMixin`

Set the diffusion coefficient for the condensing species.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L29-L46)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.set_accommodation_coefficient

*method*

```python
def set_accommodation_coefficient(accommodation_coefficient: Union[float, NDArray[np.float64]], accommodation_coefficient_units: Optional[str] = None)
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderAccommodationCoefficientMixin`

Set the dimensionless mass accommodation coefficient.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L61-L73)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.set_update_gases

*method*

```python
def set_update_gases(update_gases: bool, update_gases_units: Optional[str] = None)
```

Inherited from: `particula.dynamics.condensation.condensation_builder.condensation_builder_mixin.BuilderUpdateGasesMixin`

Set the flag controlling gas-phase updates.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/condensation_builder_mixin.py#L86-L93)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder._load_native_parameters

*method*

```python
def _load_native_parameters(parameters: dict[str, Any]) -> dict[str, Any]
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Load optional native keys and return remaining builder parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L107-L123)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder._native_strategy_kwargs

*method*

```python
def _native_strategy_kwargs() -> dict[str, Any]
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Return process configuration without constructing data or facades.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L125-L127)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.set_activity_strategy

*method*

```python
def set_activity_strategy(activity_strategy: ActivityStrategy | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Set the activity model that reads the full particle composition.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L52-L63)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.set_skip_partitioning_indices

*method*

```python
def set_skip_partitioning_indices(indices: Sequence[int] | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Detach full-gas skip indices; execution checks lane bounds.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L98-L105)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.set_species_map

*method*

```python
def set_species_map(species_map: CondensationSpeciesMap | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Set immutable full-gas provenance and explicit transfer pairs.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L39-L50)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.set_surface_strategy

*method*

```python
def set_surface_strategy(surface_strategy: SurfaceStrategy | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Set the surface model that reads the full particle composition.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L65-L74)

### particula.dynamics.condensation.condensation_builder.condensation_latent_heat_builder.CondensationLatentHeatBuilder.set_vapor_pressure_strategy

*method*

```python
def set_vapor_pressure_strategy(vapor_pressure_strategy: VaporPressureStrategy | Sequence[VaporPressureStrategy] | None) -> Self
```

Inherited from: `particula.dynamics.condensation.condensation_builder.native_configuration.BuilderNativeCondensationMixin`

Set one vector model or a detached ordered model sequence.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/condensation_builder/native_configuration.py#L76-L96)
