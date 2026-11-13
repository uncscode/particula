# particula.particles.surface_builders

*module*

Builder class for Surface strategies, for how to account for
surface tension in the calculation of the Kelvin effect.

This builds the strategy and checks that the required parameters are set,
and converts the units of the parameters if necessary.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_builders.py#L1-L1)

## particula.particles.surface_builders.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_builders.py#L28-L28)

## particula.particles.surface_builders.SurfaceStrategyMassBuilder

*class*

```python
class SurfaceStrategyMassBuilder(BuilderABC, BuilderSurfaceTensionMixin, BuilderSurfaceTensionTableMixin, BuilderTemperatureTableMixin, BuilderDensityMixin, BuilderPhaseIndexMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderSurfaceTensionMixin`, `particula.builder_mixin.BuilderSurfaceTensionTableMixin`, `particula.builder_mixin.BuilderTemperatureTableMixin`, `particula.builder_mixin.BuilderDensityMixin`, `particula.builder_mixin.BuilderPhaseIndexMixin`

Builder class for SurfaceStrategyMass objects.

For calculating the Kelvin effect based on mass mixing. Needed for
the effective surface tension calculation.

Methods:
- set_surface_tension : Set the surface tension in N/m.
- set_density : Set the density in kg/m^3.
- set_parameters : Configure multiple parameters at once.
- set_phase_index : Optionally assign species to phases.
- build : Validate parameters and return the strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_builders.py#L87-L135)

### particula.particles.surface_builders.SurfaceStrategyMassBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.particles.surface_builders.SurfaceStrategyMassBuilder.surface_tension

*attribute* · *instance attribute*

```python
surface_tension: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.builder_mixin.BuilderSurfaceTensionMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L71-L71)

### particula.particles.surface_builders.SurfaceStrategyMassBuilder.surface_tension_table

*attribute* · *instance attribute*

```python
surface_tension_table: Optional[NDArray[np.float64]] = None
```

Inherited from: `particula.builder_mixin.BuilderSurfaceTensionTableMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L110-L110)

### particula.particles.surface_builders.SurfaceStrategyMassBuilder.temperature_table

*attribute* · *instance attribute*

```python
temperature_table: Optional[NDArray[np.float64]] = None
```

Inherited from: `particula.builder_mixin.BuilderTemperatureTableMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L426-L426)

### particula.particles.surface_builders.SurfaceStrategyMassBuilder.density

*attribute* · *instance attribute*

```python
density: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.builder_mixin.BuilderDensityMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L34-L34)

### particula.particles.surface_builders.SurfaceStrategyMassBuilder.phase_index

*attribute* · *instance attribute*

```python
phase_index: Optional[NDArray[np.int_]] = None
```

Inherited from: `particula.builder_mixin.BuilderPhaseIndexMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L257-L257)

### particula.particles.surface_builders.SurfaceStrategyMassBuilder.__init__

*method*

```python
def __init__()
```

Initialize the SurfaceStrategyMassBuilder.

Sets up the builder with required parameters for creating a
SurfaceStrategyMass instance including surface tension and density.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_builders.py#L108-L120)

### particula.particles.surface_builders.SurfaceStrategyMassBuilder.build

*method*

```python
def build() -> SurfaceStrategyMass
```

Validate and return the SurfaceStrategyMass object.

**Returns**

- `SurfaceStrategyMass` (`SurfaceStrategyMass`) — Instance of the SurfaceStrategyMass object.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_builders.py#L122-L135)

### particula.particles.surface_builders.SurfaceStrategyMassBuilder.check_keys

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

### particula.particles.surface_builders.SurfaceStrategyMassBuilder.pre_build_check

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

### particula.particles.surface_builders.SurfaceStrategyMassBuilder.set_parameters

*method*

```python
def set_parameters(parameters: dict[str, Any]) -> BuilderABC
```

Inherited from: `particula.abc_builder.BuilderABC`

Set parameters from a dictionary, handling any '_units' suffix.

**Parameters**

- `parameters` (`dict[str, Any]`) — The parameters dictionary to set.

**Returns**

- `BuilderABC` (`BuilderABC`) — This builder object with the set parameters.

**Raises**

- `ValueError` — If any required key is missing.
- `Warning` — If using default units for any parameter.

**Examples**

```py
builder = Builder().set_parameters({
    "parameter1": 1,
    "parameter2": 2,
    "parameter2_units": "K",
})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L96-L130)

### particula.particles.surface_builders.SurfaceStrategyMassBuilder.set_surface_tension

*method*

```python
def set_surface_tension(surface_tension: Union[float, NDArray[np.float64]], surface_tension_units: str)
```

Inherited from: `particula.builder_mixin.BuilderSurfaceTensionMixin`

Set the particle surface tension in newtons per meter.

**Parameters**

- `surface_tension` (`Union[float, NDArray[np.float64]]`) — Positive surface tension value to store.
- `surface_tension_units` (`str`) — Units for ``surface_tension``. Defaults to ``"N/m"`` and accepts any supported conversion.

**Returns**

- `self` (`BuilderSurfaceTensionMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L73-L95)

### particula.particles.surface_builders.SurfaceStrategyMassBuilder.set_surface_tension_table

*method*

```python
def set_surface_tension_table(surface_tension_table: NDArray[np.float64], surface_tension_table_units: str = 'N/m')
```

Inherited from: `particula.builder_mixin.BuilderSurfaceTensionTableMixin`

Normalize a surface tension table to newtons per meter.

**Parameters**

- `surface_tension_table` (`NDArray[np.float64]`) — Array of surface tension values.
- `surface_tension_table_units` (`str`) (default: `'N/m'`) — Units for the values. Defaults to ``"N/m"``.

**Returns**

- `self` (`BuilderSurfaceTensionTableMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L112-L134)

### particula.particles.surface_builders.SurfaceStrategyMassBuilder.set_temperature_table

*method*

```python
def set_temperature_table(temperature_table: NDArray[np.float64], temperature_table_units: str = 'K')
```

Inherited from: `particula.builder_mixin.BuilderTemperatureTableMixin`

Assign a table of temperatures in Kelvin.

**Parameters**

- `temperature_table` (`NDArray[np.float64]`) — Array of temperature values.
- `temperature_table_units` (`str`) (default: `'K'`) — Units for the values. Defaults to ``"K"``.

**Returns**

- `self` (`BuilderTemperatureTableMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L428-L453)

### particula.particles.surface_builders.SurfaceStrategyMassBuilder.set_density

*method*

```python
def set_density(density: Union[float, NDArray[np.float64]], density_units: str)
```

Inherited from: `particula.builder_mixin.BuilderDensityMixin`

Set the particle density in kilograms per cubic meter.

**Parameters**

- `density` (`Union[float, NDArray[np.float64]]`) — Positive density value to store.
- `density_units` (`str`) — Units for ``density``. Defaults to ``"kg/m^3"`` and accepts any unit supported by the converter.

**Returns**

- `self` (`BuilderDensityMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L36-L56)

### particula.particles.surface_builders.SurfaceStrategyMassBuilder.set_phase_index

*method*

```python
def set_phase_index(phase_index: Union[Sequence[int], NDArray[np.int_]], phase_index_units: Optional[str] = None)
```

Inherited from: `particula.builder_mixin.BuilderPhaseIndexMixin`

Assign phase membership indices to species.

**Parameters**

- `phase_index` (`Union[Sequence[int], NDArray[np.int_]]`) — Sequence or array of integers describing phase IDs.
- `phase_index_units` (`Optional[str]`) (default: `None`) — Optional units that are logged and ignored.

**Returns**

- `self` (`BuilderPhaseIndexMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L259-L276)

## particula.particles.surface_builders.SurfaceStrategyMolarBuilder

*class*

```python
class SurfaceStrategyMolarBuilder(BuilderABC, BuilderDensityMixin, BuilderSurfaceTensionMixin, BuilderSurfaceTensionTableMixin, BuilderTemperatureTableMixin, BuilderMolarMassMixin, BuilderPhaseIndexMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderDensityMixin`, `particula.builder_mixin.BuilderSurfaceTensionMixin`, `particula.builder_mixin.BuilderSurfaceTensionTableMixin`, `particula.builder_mixin.BuilderTemperatureTableMixin`, `particula.builder_mixin.BuilderMolarMassMixin`, `particula.builder_mixin.BuilderPhaseIndexMixin`

Builder class for SurfaceStrategyMolar objects.

For calculating the Kelvin effect based on molar mass. Needed for
the effective surface tension calculation.

Methods:
- set_surface_tension : Set the surface tension in N/m.
- set_density : Set the density in kg/m^3.
- set_molar_mass : Set the molar mass in kg/mol.
- set_parameters : Configure multiple parameters at once.
- set_phase_index : Optionally assign species to phases.
- build : Validate parameters and return the strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_builders.py#L31-L84)

### particula.particles.surface_builders.SurfaceStrategyMolarBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.particles.surface_builders.SurfaceStrategyMolarBuilder.density

*attribute* · *instance attribute*

```python
density: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.builder_mixin.BuilderDensityMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L34-L34)

### particula.particles.surface_builders.SurfaceStrategyMolarBuilder.surface_tension

*attribute* · *instance attribute*

```python
surface_tension: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.builder_mixin.BuilderSurfaceTensionMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L71-L71)

### particula.particles.surface_builders.SurfaceStrategyMolarBuilder.surface_tension_table

*attribute* · *instance attribute*

```python
surface_tension_table: Optional[NDArray[np.float64]] = None
```

Inherited from: `particula.builder_mixin.BuilderSurfaceTensionTableMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L110-L110)

### particula.particles.surface_builders.SurfaceStrategyMolarBuilder.temperature_table

*attribute* · *instance attribute*

```python
temperature_table: Optional[NDArray[np.float64]] = None
```

Inherited from: `particula.builder_mixin.BuilderTemperatureTableMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L426-L426)

### particula.particles.surface_builders.SurfaceStrategyMolarBuilder.molar_mass

*attribute* · *instance attribute*

```python
molar_mass: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.builder_mixin.BuilderMolarMassMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L149-L149)

### particula.particles.surface_builders.SurfaceStrategyMolarBuilder.phase_index

*attribute* · *instance attribute*

```python
phase_index: Optional[NDArray[np.int_]] = None
```

Inherited from: `particula.builder_mixin.BuilderPhaseIndexMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L257-L257)

### particula.particles.surface_builders.SurfaceStrategyMolarBuilder.__init__

*method*

```python
def __init__()
```

Initialize the SurfaceStrategyMolarBuilder.

Sets up the builder with required parameters for creating a
SurfaceStrategyMolar instance including surface tension, density,
and molar mass.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_builders.py#L54-L68)

### particula.particles.surface_builders.SurfaceStrategyMolarBuilder.build

*method*

```python
def build() -> SurfaceStrategyMolar
```

Validate and return the SurfaceStrategyMolar object.

**Returns**

- `SurfaceStrategyMolar` (`SurfaceStrategyMolar`) — Instance of the SurfaceStrategyMolar object.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_builders.py#L70-L84)

### particula.particles.surface_builders.SurfaceStrategyMolarBuilder.check_keys

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

### particula.particles.surface_builders.SurfaceStrategyMolarBuilder.pre_build_check

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

### particula.particles.surface_builders.SurfaceStrategyMolarBuilder.set_parameters

*method*

```python
def set_parameters(parameters: dict[str, Any]) -> BuilderABC
```

Inherited from: `particula.abc_builder.BuilderABC`

Set parameters from a dictionary, handling any '_units' suffix.

**Parameters**

- `parameters` (`dict[str, Any]`) — The parameters dictionary to set.

**Returns**

- `BuilderABC` (`BuilderABC`) — This builder object with the set parameters.

**Raises**

- `ValueError` — If any required key is missing.
- `Warning` — If using default units for any parameter.

**Examples**

```py
builder = Builder().set_parameters({
    "parameter1": 1,
    "parameter2": 2,
    "parameter2_units": "K",
})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L96-L130)

### particula.particles.surface_builders.SurfaceStrategyMolarBuilder.set_density

*method*

```python
def set_density(density: Union[float, NDArray[np.float64]], density_units: str)
```

Inherited from: `particula.builder_mixin.BuilderDensityMixin`

Set the particle density in kilograms per cubic meter.

**Parameters**

- `density` (`Union[float, NDArray[np.float64]]`) — Positive density value to store.
- `density_units` (`str`) — Units for ``density``. Defaults to ``"kg/m^3"`` and accepts any unit supported by the converter.

**Returns**

- `self` (`BuilderDensityMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L36-L56)

### particula.particles.surface_builders.SurfaceStrategyMolarBuilder.set_surface_tension

*method*

```python
def set_surface_tension(surface_tension: Union[float, NDArray[np.float64]], surface_tension_units: str)
```

Inherited from: `particula.builder_mixin.BuilderSurfaceTensionMixin`

Set the particle surface tension in newtons per meter.

**Parameters**

- `surface_tension` (`Union[float, NDArray[np.float64]]`) — Positive surface tension value to store.
- `surface_tension_units` (`str`) — Units for ``surface_tension``. Defaults to ``"N/m"`` and accepts any supported conversion.

**Returns**

- `self` (`BuilderSurfaceTensionMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L73-L95)

### particula.particles.surface_builders.SurfaceStrategyMolarBuilder.set_surface_tension_table

*method*

```python
def set_surface_tension_table(surface_tension_table: NDArray[np.float64], surface_tension_table_units: str = 'N/m')
```

Inherited from: `particula.builder_mixin.BuilderSurfaceTensionTableMixin`

Normalize a surface tension table to newtons per meter.

**Parameters**

- `surface_tension_table` (`NDArray[np.float64]`) — Array of surface tension values.
- `surface_tension_table_units` (`str`) (default: `'N/m'`) — Units for the values. Defaults to ``"N/m"``.

**Returns**

- `self` (`BuilderSurfaceTensionTableMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L112-L134)

### particula.particles.surface_builders.SurfaceStrategyMolarBuilder.set_temperature_table

*method*

```python
def set_temperature_table(temperature_table: NDArray[np.float64], temperature_table_units: str = 'K')
```

Inherited from: `particula.builder_mixin.BuilderTemperatureTableMixin`

Assign a table of temperatures in Kelvin.

**Parameters**

- `temperature_table` (`NDArray[np.float64]`) — Array of temperature values.
- `temperature_table_units` (`str`) (default: `'K'`) — Units for the values. Defaults to ``"K"``.

**Returns**

- `self` (`BuilderTemperatureTableMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L428-L453)

### particula.particles.surface_builders.SurfaceStrategyMolarBuilder.set_molar_mass

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

### particula.particles.surface_builders.SurfaceStrategyMolarBuilder.set_phase_index

*method*

```python
def set_phase_index(phase_index: Union[Sequence[int], NDArray[np.int_]], phase_index_units: Optional[str] = None)
```

Inherited from: `particula.builder_mixin.BuilderPhaseIndexMixin`

Assign phase membership indices to species.

**Parameters**

- `phase_index` (`Union[Sequence[int], NDArray[np.int_]]`) — Sequence or array of integers describing phase IDs.
- `phase_index_units` (`Optional[str]`) (default: `None`) — Optional units that are logged and ignored.

**Returns**

- `self` (`BuilderPhaseIndexMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L259-L276)

## particula.particles.surface_builders.SurfaceStrategyVolumeBuilder

*class*

```python
class SurfaceStrategyVolumeBuilder(BuilderABC, BuilderSurfaceTensionMixin, BuilderSurfaceTensionTableMixin, BuilderTemperatureTableMixin, BuilderDensityMixin, BuilderPhaseIndexMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderSurfaceTensionMixin`, `particula.builder_mixin.BuilderSurfaceTensionTableMixin`, `particula.builder_mixin.BuilderTemperatureTableMixin`, `particula.builder_mixin.BuilderDensityMixin`, `particula.builder_mixin.BuilderPhaseIndexMixin`

Builder class for SurfaceStrategyVolume objects.

For calculating the Kelvin effect based on volume mixing. Needed for
the effective surface tension calculation.

Methods:
- set_surface_tension : Set the surface tension in N/m.
- set_density : Set the density in kg/m^3.
- set_parameters : Configure multiple parameters at once.
- set_phase_index : Optionally assign species to phases.
- build : Validate parameters and return the strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_builders.py#L138-L187)

### particula.particles.surface_builders.SurfaceStrategyVolumeBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.particles.surface_builders.SurfaceStrategyVolumeBuilder.surface_tension

*attribute* · *instance attribute*

```python
surface_tension: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.builder_mixin.BuilderSurfaceTensionMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L71-L71)

### particula.particles.surface_builders.SurfaceStrategyVolumeBuilder.surface_tension_table

*attribute* · *instance attribute*

```python
surface_tension_table: Optional[NDArray[np.float64]] = None
```

Inherited from: `particula.builder_mixin.BuilderSurfaceTensionTableMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L110-L110)

### particula.particles.surface_builders.SurfaceStrategyVolumeBuilder.temperature_table

*attribute* · *instance attribute*

```python
temperature_table: Optional[NDArray[np.float64]] = None
```

Inherited from: `particula.builder_mixin.BuilderTemperatureTableMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L426-L426)

### particula.particles.surface_builders.SurfaceStrategyVolumeBuilder.density

*attribute* · *instance attribute*

```python
density: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.builder_mixin.BuilderDensityMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L34-L34)

### particula.particles.surface_builders.SurfaceStrategyVolumeBuilder.phase_index

*attribute* · *instance attribute*

```python
phase_index: Optional[NDArray[np.int_]] = None
```

Inherited from: `particula.builder_mixin.BuilderPhaseIndexMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L257-L257)

### particula.particles.surface_builders.SurfaceStrategyVolumeBuilder.__init__

*method*

```python
def __init__()
```

Initialize the SurfaceStrategyVolumeBuilder.

Sets up the builder with required parameters for creating a
SurfaceStrategyVolume instance including surface tension and density.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_builders.py#L159-L171)

### particula.particles.surface_builders.SurfaceStrategyVolumeBuilder.build

*method*

```python
def build() -> SurfaceStrategyVolume
```

Validate and return the SurfaceStrategyVolume object.

**Returns**

- `SurfaceStrategyVolume` (`SurfaceStrategyVolume`) — Instance of the SurfaceStrategyVolume object.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_builders.py#L173-L187)

### particula.particles.surface_builders.SurfaceStrategyVolumeBuilder.check_keys

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

### particula.particles.surface_builders.SurfaceStrategyVolumeBuilder.pre_build_check

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

### particula.particles.surface_builders.SurfaceStrategyVolumeBuilder.set_parameters

*method*

```python
def set_parameters(parameters: dict[str, Any]) -> BuilderABC
```

Inherited from: `particula.abc_builder.BuilderABC`

Set parameters from a dictionary, handling any '_units' suffix.

**Parameters**

- `parameters` (`dict[str, Any]`) — The parameters dictionary to set.

**Returns**

- `BuilderABC` (`BuilderABC`) — This builder object with the set parameters.

**Raises**

- `ValueError` — If any required key is missing.
- `Warning` — If using default units for any parameter.

**Examples**

```py
builder = Builder().set_parameters({
    "parameter1": 1,
    "parameter2": 2,
    "parameter2_units": "K",
})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L96-L130)

### particula.particles.surface_builders.SurfaceStrategyVolumeBuilder.set_surface_tension

*method*

```python
def set_surface_tension(surface_tension: Union[float, NDArray[np.float64]], surface_tension_units: str)
```

Inherited from: `particula.builder_mixin.BuilderSurfaceTensionMixin`

Set the particle surface tension in newtons per meter.

**Parameters**

- `surface_tension` (`Union[float, NDArray[np.float64]]`) — Positive surface tension value to store.
- `surface_tension_units` (`str`) — Units for ``surface_tension``. Defaults to ``"N/m"`` and accepts any supported conversion.

**Returns**

- `self` (`BuilderSurfaceTensionMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L73-L95)

### particula.particles.surface_builders.SurfaceStrategyVolumeBuilder.set_surface_tension_table

*method*

```python
def set_surface_tension_table(surface_tension_table: NDArray[np.float64], surface_tension_table_units: str = 'N/m')
```

Inherited from: `particula.builder_mixin.BuilderSurfaceTensionTableMixin`

Normalize a surface tension table to newtons per meter.

**Parameters**

- `surface_tension_table` (`NDArray[np.float64]`) — Array of surface tension values.
- `surface_tension_table_units` (`str`) (default: `'N/m'`) — Units for the values. Defaults to ``"N/m"``.

**Returns**

- `self` (`BuilderSurfaceTensionTableMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L112-L134)

### particula.particles.surface_builders.SurfaceStrategyVolumeBuilder.set_temperature_table

*method*

```python
def set_temperature_table(temperature_table: NDArray[np.float64], temperature_table_units: str = 'K')
```

Inherited from: `particula.builder_mixin.BuilderTemperatureTableMixin`

Assign a table of temperatures in Kelvin.

**Parameters**

- `temperature_table` (`NDArray[np.float64]`) — Array of temperature values.
- `temperature_table_units` (`str`) (default: `'K'`) — Units for the values. Defaults to ``"K"``.

**Returns**

- `self` (`BuilderTemperatureTableMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L428-L453)

### particula.particles.surface_builders.SurfaceStrategyVolumeBuilder.set_density

*method*

```python
def set_density(density: Union[float, NDArray[np.float64]], density_units: str)
```

Inherited from: `particula.builder_mixin.BuilderDensityMixin`

Set the particle density in kilograms per cubic meter.

**Parameters**

- `density` (`Union[float, NDArray[np.float64]]`) — Positive density value to store.
- `density_units` (`str`) — Units for ``density``. Defaults to ``"kg/m^3"`` and accepts any unit supported by the converter.

**Returns**

- `self` (`BuilderDensityMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L36-L56)

### particula.particles.surface_builders.SurfaceStrategyVolumeBuilder.set_phase_index

*method*

```python
def set_phase_index(phase_index: Union[Sequence[int], NDArray[np.int_]], phase_index_units: Optional[str] = None)
```

Inherited from: `particula.builder_mixin.BuilderPhaseIndexMixin`

Assign phase membership indices to species.

**Parameters**

- `phase_index` (`Union[Sequence[int], NDArray[np.int_]]`) — Sequence or array of integers describing phase IDs.
- `phase_index_units` (`Optional[str]`) (default: `None`) — Optional units that are logged and ignored.

**Returns**

- `self` (`BuilderPhaseIndexMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L259-L276)
