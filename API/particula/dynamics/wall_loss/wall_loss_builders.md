# particula.dynamics.wall_loss.wall_loss_builders

*module*

Wall loss strategy builders.

Provides builder classes for constructing wall loss strategies with validated
parameters and unit conversion support.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_builders.py#L1-L1)

## particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder

*class*

```python
class ChargedWallLossBuilder(BuilderABC, BuilderWallEddyDiffusivityMixin, BuilderDistributionTypeMixin, BuilderChamberRadiusMixin, BuilderChamberDimensionsMixin, BuilderWallPotentialMixin, BuilderWallElectricFieldMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderWallEddyDiffusivityMixin`, `particula.builder_mixin.BuilderDistributionTypeMixin`, `particula.builder_mixin.BuilderChamberRadiusMixin`, `particula.builder_mixin.BuilderChamberDimensionsMixin`, `particula.builder_mixin.BuilderWallPotentialMixin`, `particula.builder_mixin.BuilderWallElectricFieldMixin`

Builder for charged wall loss strategies.

Configures image-charge-enhanced wall loss with optional
electric-field drift for spherical or rectangular chambers.
Requires geometry plus the matching size parameter. Zero wall potential
still allows charge-driven enhancement.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_builders.py#L126-L282)

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder.chamber_geometry

*attribute* · *instance attribute*

```python
chamber_geometry: Optional[str] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_builders.py#L155-L155)

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder.wall_eddy_diffusivity

*attribute* · *instance attribute*

```python
wall_eddy_diffusivity: Optional[float] = None
```

Inherited from: `particula.builder_mixin.BuilderWallEddyDiffusivityMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L616-L616)

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder.distribution_type

*attribute* · *instance attribute*

```python
distribution_type: str = 'discrete'
```

Inherited from: `particula.builder_mixin.BuilderDistributionTypeMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L944-L944)

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder.chamber_radius

*attribute* · *instance attribute*

```python
chamber_radius: Optional[float] = None
```

Inherited from: `particula.builder_mixin.BuilderChamberRadiusMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L663-L663)

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder.chamber_dimensions

*attribute* · *instance attribute*

```python
chamber_dimensions: Optional[Tuple[float, float, float]] = None
```

Inherited from: `particula.builder_mixin.BuilderChamberDimensionsMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L712-L712)

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder.wall_potential

*attribute* · *instance attribute*

```python
wall_potential: float = 0.0
```

Inherited from: `particula.builder_mixin.BuilderWallPotentialMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L795-L795)

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder.wall_electric_field

*attribute* · *instance attribute*

```python
wall_electric_field: Union[float, Tuple[float, float, float]] = 0.0
```

Inherited from: `particula.builder_mixin.BuilderWallElectricFieldMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L837-L837)

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder.__init__

*method*

```python
def __init__()
```

Initialize the charged wall loss builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_builders.py#L143-L155)

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder.build

*method*

```python
def build() -> ChargedWallLossStrategy
```

Build and return a charged wall loss strategy.

**Returns**

- `ChargedWallLossStrategy` (`ChargedWallLossStrategy`) — Charged strategy with geometry-specific sizing and electrostatic settings.

**Raises**

- `ValueError` — If geometry is unset, matching size parameters are
missing, or required values are not provided.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_builders.py#L243-L282)

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder.set_chamber_geometry

*method*

```python
def set_chamber_geometry(chamber_geometry: str)
```

Set chamber geometry.

**Parameters**

- `chamber_geometry` (`str`) — ``"spherical"`` or ``"rectangular"``.

**Returns**

- `result` (`ChargedWallLossBuilder`) — This instance for method chaining.

**Raises**

- `ValueError` — If ``chamber_geometry`` is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_builders.py#L157-L175)

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder.set_parameters

*method*

```python
def set_parameters(parameters: dict)
```

Set required and optional parameters from mapping.

**Parameters**

- `parameters` (`dict`) — Mapping containing required ``wall_eddy_diffusivity`` and ``chamber_geometry`` plus optional geometry size, ``wall_potential``, and ``wall_electric_field`` entries with optional ``*_units`` keys.

**Returns**

- `result` (`ChargedWallLossBuilder`) — This instance for method chaining.

**Raises**

- `ValueError` — If required keys are missing or unexpected keys are
provided.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_builders.py#L177-L241)

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder.check_keys

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

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder.pre_build_check

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

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder.set_wall_eddy_diffusivity

*method*

```python
def set_wall_eddy_diffusivity(wall_eddy_diffusivity: float, wall_eddy_diffusivity_units: str = '1/s') -> BuilderWallEddyDiffusivityMixin
```

Inherited from: `particula.builder_mixin.BuilderWallEddyDiffusivityMixin`

Set wall eddy diffusivity in 1/s.

**Parameters**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity value.
- `wall_eddy_diffusivity_units` (`str`) (default: `'1/s'`) — Units for the value. Defaults to "1/s".

**Returns**

- `self` (`BuilderWallEddyDiffusivityMixin`) — Self for method chaining.

**Raises**

- `ValueError` — If ``wall_eddy_diffusivity`` is not positive.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L618-L648)

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder.set_distribution_type

*method*

```python
def set_distribution_type(distribution_type: str) -> Self
```

Inherited from: `particula.builder_mixin.BuilderDistributionTypeMixin`

Set the distribution type.

**Parameters**

- `distribution_type` (`str`) — One of "discrete", "continuous_pdf", or "particle_resolved".

**Returns**

- `self` (`BuilderDistributionTypeMixin`) — Self for method chaining.

**Raises**

- `ValueError` — If ``distribution_type`` is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L946-L966)

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder.set_chamber_radius

*method*

```python
def set_chamber_radius(chamber_radius: float, chamber_radius_units: str = 'm') -> BuilderChamberRadiusMixin
```

Inherited from: `particula.builder_mixin.BuilderChamberRadiusMixin`

Set the chamber radius in meters.

**Parameters**

- `chamber_radius` (`float`) — Chamber radius value.
- `chamber_radius_units` (`str`) (default: `'m'`) — Units for the value. Defaults to "m".

**Returns**

- `self` (`BuilderChamberRadiusMixin`) — Self for method chaining.

**Raises**

- `ValueError` — If ``chamber_radius`` is not positive.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L665-L697)

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder.set_chamber_dimensions

*method*

```python
def set_chamber_dimensions(chamber_dimensions: Tuple[float, float, float], chamber_dimensions_units: str = 'm') -> BuilderChamberDimensionsMixin
```

Inherited from: `particula.builder_mixin.BuilderChamberDimensionsMixin`

Set rectangular chamber dimensions in meters.

**Parameters**

- `chamber_dimensions` (`Tuple[float, float, float]`) — Tuple of (length, width, height).
- `chamber_dimensions_units` (`str`) (default: `'m'`) — Units for the values. Defaults to "m".

**Returns**

- `self` (`BuilderChamberDimensionsMixin`) — Self for method chaining.

**Raises**

- `ValueError` — If ``chamber_dimensions`` is not length three or any
value is non-positive.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L714-L779)

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder.set_wall_potential

*method*

```python
def set_wall_potential(wall_potential: float, wall_potential_units: str = 'V') -> BuilderWallPotentialMixin
```

Inherited from: `particula.builder_mixin.BuilderWallPotentialMixin`

Set wall potential in volts.

**Parameters**

- `wall_potential` (`float`) — Electrostatic wall potential in volts. Zero keeps the chamber neutral but still allows image-charge effects for charged particles.
- `wall_potential_units` (`str`) (default: `'V'`) — Units for ``wall_potential``. Defaults to "V".

**Returns**

- `self` (`BuilderWallPotentialMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L797-L820)

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder._validate_wall_electric_field

*method*

```python
def _validate_wall_electric_field(wall_electric_field: Union[float, Tuple[float, float, float]]) -> Union[float, Tuple[float, float, float]]
```

Inherited from: `particula.builder_mixin.BuilderWallElectricFieldMixin`

Validate and normalize wall electric field inputs.

**Parameters**

- `wall_electric_field` (`Union[float, Tuple[float, float, float]]`) — Scalar magnitude or ``(Ex, Ey, Ez)`` tuple.

**Returns**

- (`Union[float, Tuple[float, float, float]]`) — Union[float, Tuple[float, float, float]]: Validated scalar or tuple.

**Raises**

- `ValueError` — If tuple length is not three or entries are not finite.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L839-L864)

### particula.dynamics.wall_loss.wall_loss_builders.ChargedWallLossBuilder.set_wall_electric_field

*method*

```python
def set_wall_electric_field(wall_electric_field: Union[float, Tuple[float, float, float]], wall_electric_field_units: str = 'V/m') -> BuilderWallElectricFieldMixin
```

Inherited from: `particula.builder_mixin.BuilderWallElectricFieldMixin`

Set wall electric field magnitude in V/m.

**Parameters**

- `wall_electric_field` (`Union[float, Tuple[float, float, float]]`) — Scalar magnitude for spherical chambers or a three-element ``(Ex, Ey, Ez)`` tuple for rectangular chambers. Use ``0.0`` to disable field-driven drift.
- `wall_electric_field_units` (`str`) (default: `'V/m'`) — Units for ``wall_electric_field``. Defaults to "V/m".

**Returns**

- `self` (`BuilderWallElectricFieldMixin`) — Self for method chaining.

**Raises**

- `ValueError` — If the tuple is not length three or any entry is not
finite.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L866-L929)

## particula.dynamics.wall_loss.wall_loss_builders.RectangularWallLossBuilder

*class*

```python
class RectangularWallLossBuilder(BuilderABC, BuilderWallEddyDiffusivityMixin, BuilderChamberDimensionsMixin, BuilderDistributionTypeMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderWallEddyDiffusivityMixin`, `particula.builder_mixin.BuilderChamberDimensionsMixin`, `particula.builder_mixin.BuilderDistributionTypeMixin`

Builder for rectangular wall loss strategies.

Validates wall eddy diffusivity, chamber dimensions, and distribution type
before constructing :class:`RectangularWallLossStrategy`.

**Attributes**

- `wall_eddy_diffusivity` (`Optional[float]`) — Wall eddy diffusivity in 1/s.
- `chamber_dimensions` (`Optional[Tuple[float, float, float]]`) — Tuple of (length, width, height) in meters.
- `distribution_type` (`str`) — Distribution type for the strategy output.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_builders.py#L74-L123)

### particula.dynamics.wall_loss.wall_loss_builders.RectangularWallLossBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.wall_loss.wall_loss_builders.RectangularWallLossBuilder.wall_eddy_diffusivity

*attribute* · *instance attribute*

```python
wall_eddy_diffusivity: Optional[float] = None
```

Inherited from: `particula.builder_mixin.BuilderWallEddyDiffusivityMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L616-L616)

### particula.dynamics.wall_loss.wall_loss_builders.RectangularWallLossBuilder.chamber_dimensions

*attribute* · *instance attribute*

```python
chamber_dimensions: Optional[Tuple[float, float, float]] = None
```

Inherited from: `particula.builder_mixin.BuilderChamberDimensionsMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L712-L712)

### particula.dynamics.wall_loss.wall_loss_builders.RectangularWallLossBuilder.distribution_type

*attribute* · *instance attribute*

```python
distribution_type: str = 'discrete'
```

Inherited from: `particula.builder_mixin.BuilderDistributionTypeMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L944-L944)

### particula.dynamics.wall_loss.wall_loss_builders.RectangularWallLossBuilder.__init__

*method*

```python
def __init__()
```

Initialize the rectangular wall loss builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_builders.py#L91-L102)

### particula.dynamics.wall_loss.wall_loss_builders.RectangularWallLossBuilder.build

*method*

```python
def build() -> RectangularWallLossStrategy
```

Build and return the rectangular wall loss strategy.

**Returns**

- `RectangularWallLossStrategy` (`RectangularWallLossStrategy`) — Configured wall loss strategy.

**Raises**

- `ValueError` — If required parameters are missing or invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_builders.py#L104-L123)

### particula.dynamics.wall_loss.wall_loss_builders.RectangularWallLossBuilder.check_keys

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

### particula.dynamics.wall_loss.wall_loss_builders.RectangularWallLossBuilder.pre_build_check

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

### particula.dynamics.wall_loss.wall_loss_builders.RectangularWallLossBuilder.set_parameters

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

### particula.dynamics.wall_loss.wall_loss_builders.RectangularWallLossBuilder.set_wall_eddy_diffusivity

*method*

```python
def set_wall_eddy_diffusivity(wall_eddy_diffusivity: float, wall_eddy_diffusivity_units: str = '1/s') -> BuilderWallEddyDiffusivityMixin
```

Inherited from: `particula.builder_mixin.BuilderWallEddyDiffusivityMixin`

Set wall eddy diffusivity in 1/s.

**Parameters**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity value.
- `wall_eddy_diffusivity_units` (`str`) (default: `'1/s'`) — Units for the value. Defaults to "1/s".

**Returns**

- `self` (`BuilderWallEddyDiffusivityMixin`) — Self for method chaining.

**Raises**

- `ValueError` — If ``wall_eddy_diffusivity`` is not positive.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L618-L648)

### particula.dynamics.wall_loss.wall_loss_builders.RectangularWallLossBuilder.set_chamber_dimensions

*method*

```python
def set_chamber_dimensions(chamber_dimensions: Tuple[float, float, float], chamber_dimensions_units: str = 'm') -> BuilderChamberDimensionsMixin
```

Inherited from: `particula.builder_mixin.BuilderChamberDimensionsMixin`

Set rectangular chamber dimensions in meters.

**Parameters**

- `chamber_dimensions` (`Tuple[float, float, float]`) — Tuple of (length, width, height).
- `chamber_dimensions_units` (`str`) (default: `'m'`) — Units for the values. Defaults to "m".

**Returns**

- `self` (`BuilderChamberDimensionsMixin`) — Self for method chaining.

**Raises**

- `ValueError` — If ``chamber_dimensions`` is not length three or any
value is non-positive.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L714-L779)

### particula.dynamics.wall_loss.wall_loss_builders.RectangularWallLossBuilder.set_distribution_type

*method*

```python
def set_distribution_type(distribution_type: str) -> Self
```

Inherited from: `particula.builder_mixin.BuilderDistributionTypeMixin`

Set the distribution type.

**Parameters**

- `distribution_type` (`str`) — One of "discrete", "continuous_pdf", or "particle_resolved".

**Returns**

- `self` (`BuilderDistributionTypeMixin`) — Self for method chaining.

**Raises**

- `ValueError` — If ``distribution_type`` is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L946-L966)

## particula.dynamics.wall_loss.wall_loss_builders.SphericalWallLossBuilder

*class*

```python
class SphericalWallLossBuilder(BuilderABC, BuilderWallEddyDiffusivityMixin, BuilderChamberRadiusMixin, BuilderDistributionTypeMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderWallEddyDiffusivityMixin`, `particula.builder_mixin.BuilderChamberRadiusMixin`, `particula.builder_mixin.BuilderDistributionTypeMixin`

Builder for spherical wall loss strategies.

Validates wall eddy diffusivity, chamber radius, and distribution type
before constructing :class:`SphericalWallLossStrategy`.

**Attributes**

- `wall_eddy_diffusivity` (`Optional[float]`) — Wall eddy diffusivity in 1/s.
- `chamber_radius` (`Optional[float]`) — Chamber radius in meters.
- `distribution_type` (`str`) — Distribution type for the strategy output.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_builders.py#L25-L71)

### particula.dynamics.wall_loss.wall_loss_builders.SphericalWallLossBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.wall_loss.wall_loss_builders.SphericalWallLossBuilder.wall_eddy_diffusivity

*attribute* · *instance attribute*

```python
wall_eddy_diffusivity: Optional[float] = None
```

Inherited from: `particula.builder_mixin.BuilderWallEddyDiffusivityMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L616-L616)

### particula.dynamics.wall_loss.wall_loss_builders.SphericalWallLossBuilder.chamber_radius

*attribute* · *instance attribute*

```python
chamber_radius: Optional[float] = None
```

Inherited from: `particula.builder_mixin.BuilderChamberRadiusMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L663-L663)

### particula.dynamics.wall_loss.wall_loss_builders.SphericalWallLossBuilder.distribution_type

*attribute* · *instance attribute*

```python
distribution_type: str = 'discrete'
```

Inherited from: `particula.builder_mixin.BuilderDistributionTypeMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L944-L944)

### particula.dynamics.wall_loss.wall_loss_builders.SphericalWallLossBuilder.__init__

*method*

```python
def __init__()
```

Initialize the spherical wall loss builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_builders.py#L42-L50)

### particula.dynamics.wall_loss.wall_loss_builders.SphericalWallLossBuilder.build

*method*

```python
def build() -> SphericalWallLossStrategy
```

Build and return the spherical wall loss strategy.

**Returns**

- `SphericalWallLossStrategy` (`SphericalWallLossStrategy`) — Configured wall loss strategy.

**Raises**

- `ValueError` — If required parameters are missing or invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_builders.py#L52-L71)

### particula.dynamics.wall_loss.wall_loss_builders.SphericalWallLossBuilder.check_keys

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

### particula.dynamics.wall_loss.wall_loss_builders.SphericalWallLossBuilder.pre_build_check

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

### particula.dynamics.wall_loss.wall_loss_builders.SphericalWallLossBuilder.set_parameters

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

### particula.dynamics.wall_loss.wall_loss_builders.SphericalWallLossBuilder.set_wall_eddy_diffusivity

*method*

```python
def set_wall_eddy_diffusivity(wall_eddy_diffusivity: float, wall_eddy_diffusivity_units: str = '1/s') -> BuilderWallEddyDiffusivityMixin
```

Inherited from: `particula.builder_mixin.BuilderWallEddyDiffusivityMixin`

Set wall eddy diffusivity in 1/s.

**Parameters**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity value.
- `wall_eddy_diffusivity_units` (`str`) (default: `'1/s'`) — Units for the value. Defaults to "1/s".

**Returns**

- `self` (`BuilderWallEddyDiffusivityMixin`) — Self for method chaining.

**Raises**

- `ValueError` — If ``wall_eddy_diffusivity`` is not positive.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L618-L648)

### particula.dynamics.wall_loss.wall_loss_builders.SphericalWallLossBuilder.set_chamber_radius

*method*

```python
def set_chamber_radius(chamber_radius: float, chamber_radius_units: str = 'm') -> BuilderChamberRadiusMixin
```

Inherited from: `particula.builder_mixin.BuilderChamberRadiusMixin`

Set the chamber radius in meters.

**Parameters**

- `chamber_radius` (`float`) — Chamber radius value.
- `chamber_radius_units` (`str`) (default: `'m'`) — Units for the value. Defaults to "m".

**Returns**

- `self` (`BuilderChamberRadiusMixin`) — Self for method chaining.

**Raises**

- `ValueError` — If ``chamber_radius`` is not positive.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L665-L697)

### particula.dynamics.wall_loss.wall_loss_builders.SphericalWallLossBuilder.set_distribution_type

*method*

```python
def set_distribution_type(distribution_type: str) -> Self
```

Inherited from: `particula.builder_mixin.BuilderDistributionTypeMixin`

Set the distribution type.

**Parameters**

- `distribution_type` (`str`) — One of "discrete", "continuous_pdf", or "particle_resolved".

**Returns**

- `self` (`BuilderDistributionTypeMixin`) — Self for method chaining.

**Raises**

- `ValueError` — If ``distribution_type`` is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L946-L966)
