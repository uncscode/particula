# particula.dynamics.wall_loss

*module*

Wall loss utilities, strategies, builders, and factory.

Provides rate calculators, strategy classes, and builders for particle wall
loss in spherical and rectangular chambers. Includes charged wall loss with
image-charge enhancement even at zero wall potential, optional electric-field
drift, and deterministic or particle-resolved helpers.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/__init__.py#L1-L1)

## particula.dynamics.wall_loss.ChargedWallLossBuilder

*class*

```python
class ChargedWallLossBuilder(BuilderABC, BuilderWallEddyDiffusivityMixin, BuilderDistributionTypeMixin, BuilderChamberRadiusMixin, BuilderChamberDimensionsMixin, BuilderWallPotentialMixin, BuilderWallElectricFieldMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderWallEddyDiffusivityMixin`, `particula.builder_mixin.BuilderDistributionTypeMixin`, `particula.builder_mixin.BuilderChamberRadiusMixin`, `particula.builder_mixin.BuilderChamberDimensionsMixin`, `particula.builder_mixin.BuilderWallPotentialMixin`, `particula.builder_mixin.BuilderWallElectricFieldMixin`

Re-exported from: `particula.dynamics.wall_loss.wall_loss_builders`

Builder for charged wall loss strategies.

Configures image-charge-enhanced wall loss with optional
electric-field drift for spherical or rectangular chambers.
Requires geometry plus the matching size parameter. Zero wall potential
still allows charge-driven enhancement.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_builders.py#L126-L282)

### particula.dynamics.wall_loss.ChargedWallLossBuilder.chamber_geometry

*attribute* · *instance attribute*

```python
chamber_geometry: Optional[str] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_builders.py#L155-L155)

### particula.dynamics.wall_loss.ChargedWallLossBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.wall_loss.ChargedWallLossBuilder.wall_eddy_diffusivity

*attribute* · *instance attribute*

```python
wall_eddy_diffusivity: Optional[float] = None
```

Inherited from: `particula.builder_mixin.BuilderWallEddyDiffusivityMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L616-L616)

### particula.dynamics.wall_loss.ChargedWallLossBuilder.distribution_type

*attribute* · *instance attribute*

```python
distribution_type: str = 'discrete'
```

Inherited from: `particula.builder_mixin.BuilderDistributionTypeMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L944-L944)

### particula.dynamics.wall_loss.ChargedWallLossBuilder.chamber_radius

*attribute* · *instance attribute*

```python
chamber_radius: Optional[float] = None
```

Inherited from: `particula.builder_mixin.BuilderChamberRadiusMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L663-L663)

### particula.dynamics.wall_loss.ChargedWallLossBuilder.chamber_dimensions

*attribute* · *instance attribute*

```python
chamber_dimensions: Optional[Tuple[float, float, float]] = None
```

Inherited from: `particula.builder_mixin.BuilderChamberDimensionsMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L712-L712)

### particula.dynamics.wall_loss.ChargedWallLossBuilder.wall_potential

*attribute* · *instance attribute*

```python
wall_potential: float = 0.0
```

Inherited from: `particula.builder_mixin.BuilderWallPotentialMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L795-L795)

### particula.dynamics.wall_loss.ChargedWallLossBuilder.wall_electric_field

*attribute* · *instance attribute*

```python
wall_electric_field: Union[float, Tuple[float, float, float]] = 0.0
```

Inherited from: `particula.builder_mixin.BuilderWallElectricFieldMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L837-L837)

### particula.dynamics.wall_loss.ChargedWallLossBuilder.__init__

*method*

```python
def __init__()
```

Initialize the charged wall loss builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_builders.py#L143-L155)

### particula.dynamics.wall_loss.ChargedWallLossBuilder.build

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

### particula.dynamics.wall_loss.ChargedWallLossBuilder.set_chamber_geometry

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

### particula.dynamics.wall_loss.ChargedWallLossBuilder.set_parameters

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

### particula.dynamics.wall_loss.ChargedWallLossBuilder.check_keys

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

### particula.dynamics.wall_loss.ChargedWallLossBuilder.pre_build_check

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

### particula.dynamics.wall_loss.ChargedWallLossBuilder.set_wall_eddy_diffusivity

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

### particula.dynamics.wall_loss.ChargedWallLossBuilder.set_distribution_type

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

### particula.dynamics.wall_loss.ChargedWallLossBuilder.set_chamber_radius

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

### particula.dynamics.wall_loss.ChargedWallLossBuilder.set_chamber_dimensions

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

### particula.dynamics.wall_loss.ChargedWallLossBuilder.set_wall_potential

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

### particula.dynamics.wall_loss.ChargedWallLossBuilder._validate_wall_electric_field

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

### particula.dynamics.wall_loss.ChargedWallLossBuilder.set_wall_electric_field

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

## particula.dynamics.wall_loss.ChargedWallLossStrategy

*class*

```python
class ChargedWallLossStrategy(WallLossStrategy)
```

Bases: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Re-exported from: `particula.dynamics.wall_loss.wall_loss_strategies`

Wall loss strategy with electrostatic effects.

Extends neutral wall loss with image-charge enhancement, optional
electric-field drift, and diffusion modification for charged particles.
Behaves as the neutral strategy when both charge and field terms are zero
while still applying the image-charge-only enhancement when charge is
non-zero and ``wall_potential`` is zero.

> **References**
>
> McMurry, P. H., & Rader, D. J. (1985). Aerosol wall losses in
> electrically charged chambers. Aerosol Science and Technology, 4(3),
> 249–268.
> Lai, A. C. K., & Nazaroff, W. W. (2000). Modeling indoor particle
> deposition from turbulent flow onto smooth surfaces. J. Aerosol Sci.,
> 31(4), 463–476.
> Hinds, W. C. (1999). *Aerosol Technology*. Wiley.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L411-L812)

### particula.dynamics.wall_loss.ChargedWallLossStrategy.chamber_dimensions

*attribute* · *instance attribute*

```python
chamber_dimensions: Union[Tuple[float, float, float], None]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L432-L432)

### particula.dynamics.wall_loss.ChargedWallLossStrategy.chamber_geometry

*attribute* · *instance attribute*

```python
chamber_geometry: str = geometry
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L503-L503)

### particula.dynamics.wall_loss.ChargedWallLossStrategy.chamber_radius

*attribute* · *instance attribute*

```python
chamber_radius: Union[float, None]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L431-L431)

### particula.dynamics.wall_loss.ChargedWallLossStrategy.wall_electric_field

*attribute* · *instance attribute*

```python
wall_electric_field: Union[float, Tuple[float, float, float]] = wall_electric_field
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L505-L505)

### particula.dynamics.wall_loss.ChargedWallLossStrategy.wall_potential

*attribute* · *instance attribute*

```python
wall_potential: float = float(wall_potential)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L504-L504)

### particula.dynamics.wall_loss.ChargedWallLossStrategy.distribution_type

*attribute* · *instance attribute*

```python
distribution_type: str = distribution_type
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L176-L176)

### particula.dynamics.wall_loss.ChargedWallLossStrategy.random_generator

*attribute* · *instance attribute*

```python
random_generator: np.random.Generator = np.random.default_rng()
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L177-L177)

### particula.dynamics.wall_loss.ChargedWallLossStrategy.wall_eddy_diffusivity

*attribute* · *instance attribute*

```python
wall_eddy_diffusivity: float = wall_eddy_diffusivity
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L175-L175)

### particula.dynamics.wall_loss.ChargedWallLossStrategy._geometry_scale

*property*

```python
_geometry_scale: float
```

Return characteristic geometry length for field scaling.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L508-L514)

### particula.dynamics.wall_loss.ChargedWallLossStrategy.__init__

*method*

```python
def __init__(wall_eddy_diffusivity: float, chamber_geometry: str, chamber_radius: Union[float, None] = None, chamber_dimensions: Union[Tuple[float, float, float], None] = None, wall_potential: float = 0.0, wall_electric_field: Union[float, Tuple[float, float, float]] = 0.0, distribution_type: str = 'discrete') -> None
```

Initialize charged wall loss strategy.

**Parameters**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity [1/s].
- `chamber_geometry` (`str`) — Geometry string ("spherical" or "rectangular").
- `chamber_radius` (`Union[float, None]`) (default: `None`) — Radius for spherical chambers [m].
- `chamber_dimensions` (`Union[Tuple[float, float, float], None]`) (default: `None`) — Dimensions (length, width, height) for rectangular chambers [m].
- `wall_potential` (`float`) (default: `0.0`) — Wall potential [V]; zero keeps image-charge term.
- `wall_electric_field` (`Union[float, Tuple[float, float, float]]`) (default: `0.0`) — Electric field magnitude [V/m] (scalar) or 3-vector for rectangular chambers. Zero disables drift term.
- `distribution_type` (`str`) (default: `'discrete'`) — Distribution type.

**Raises**

- `ValueError` — If geometry is invalid or required geometry parameters
are missing.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L436-L505)

### particula.dynamics.wall_loss.ChargedWallLossStrategy._combine_coefficients

*method*

```python
def _combine_coefficients(neutral: NDArray[np.float64], electrostatic_factor: NDArray[np.float64], drift_term: NDArray[np.float64]) -> NDArray[np.float64]
```

Combine neutral, electrostatic, and drift contributions.

**Parameters**

- `neutral` (`NDArray[np.float64]`) — Neutral wall loss coefficient in 1/s.
- `electrostatic_factor` (`NDArray[np.float64]`) — Multiplicative factor from image charge.
- `drift_term` (`NDArray[np.float64]`) — Additive drift term in 1/s.

**Returns**

- (`NDArray[np.float64]`) — Combined wall loss coefficient clipped to non-negative finite
- (`NDArray[np.float64]`) — values.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L650-L672)

### particula.dynamics.wall_loss.ChargedWallLossStrategy._drift_term

*method*

```python
def _drift_term(particle_radius: NDArray[np.float64], particle_charge: NDArray[np.float64], temperature: float, pressure: float) -> NDArray[np.float64]
```

Compute electric-field drift contribution.

Returns zero when charge is zero or when the resolved electric field
is zero.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Particle radii in meters.
- `particle_charge` (`NDArray[np.float64]`) — Particle charge in elementary charges.
- `temperature` (`float`) — Gas temperature in kelvin.
- `pressure` (`float`) — Gas pressure in pascals.

**Returns**

- (`NDArray[np.float64]`) — Additive drift term in 1/s for the wall loss coefficient.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L612-L648)

### particula.dynamics.wall_loss.ChargedWallLossStrategy._electrostatic_factor

*method*

```python
def _electrostatic_factor(particle_radius: NDArray[np.float64], particle_charge: NDArray[np.float64], temperature: float) -> NDArray[np.float64]
```

Compute electrostatic enhancement factor.

Applies image-charge-derived enhancement based on particle charge.
Returns ones when charge is zero so the strategy reduces to neutral
even if ``wall_potential`` is zero.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Particle radii in meters.
- `particle_charge` (`NDArray[np.float64]`) — Particle charge in elementary charges.
- `temperature` (`float`) — Gas temperature in kelvin.

**Returns**

- (`NDArray[np.float64]`) — Multiplicative electrostatic factor for the wall loss coefficient.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L558-L591)

### particula.dynamics.wall_loss.ChargedWallLossStrategy._neutral_coefficient

*method*

```python
def _neutral_coefficient(particle_radius: NDArray[np.float64], particle_density: NDArray[np.float64], temperature: float, pressure: float) -> NDArray[np.float64]
```

Return neutral wall loss coefficient for configured geometry.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Particle radii in meters.
- `particle_density` (`NDArray[np.float64]`) — Particle densities in kg/m³.
- `temperature` (`float`) — Gas temperature in kelvin.
- `pressure` (`float`) — Gas pressure in pascals.

**Returns**

- (`NDArray[np.float64]`) — Neutral wall loss coefficient in 1/s for each particle size.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L516-L556)

### particula.dynamics.wall_loss.ChargedWallLossStrategy._resolve_electric_field

*method*

```python
def _resolve_electric_field() -> float
```

Return resolved electric field magnitude in V/m.

Combines the configured wall electric field (scalar or vector) with a
geometry-scaled potential-derived field; returns zero when both are
zero.

**Returns**

- (`float`) — Total electric field magnitude in volts per meter.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L593-L610)

### particula.dynamics.wall_loss.ChargedWallLossStrategy.compute_coefficient_from_arrays

*method*

```python
def compute_coefficient_from_arrays(particle_radius: NDArray[np.float64], particle_density: NDArray[np.float64], particle_charge: NDArray[np.float64], temperature: float, pressure: float) -> NDArray[np.float64]
```

Compute charged wall loss coefficient from explicit arrays.

Public method for computing coefficients with explicit charge arrays,
useful for rate helpers and direct calculations without cached state.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Particle radii in meters.
- `particle_density` (`NDArray[np.float64]`) — Particle densities in kg/m³.
- `particle_charge` (`NDArray[np.float64]`) — Particle charge in elementary charges.
- `temperature` (`float`) — Gas temperature in kelvin.
- `pressure` (`float`) — Gas pressure in pascals.

**Returns**

- (`NDArray[np.float64]`) — Wall loss coefficient in 1/s for each particle.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L744-L788)

### particula.dynamics.wall_loss.ChargedWallLossStrategy.loss_coefficient

*method*

```python
def loss_coefficient(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Compute charged wall loss coefficient from particle properties.

Applies image-charge enhancement even when ``wall_potential`` is zero
and adds drift when ``wall_electric_field`` is non-zero.
Reduces to the neutral coefficient when charge and field are zero and
supports particle-resolved active subsets.

**Parameters**

- `particle` (`ParticleData`) — Native particle data providing radius, density, and charge.
- `temperature` (`float`) — Gas temperature in kelvin.
- `pressure` (`float`) — Gas pressure in pascals.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Wall loss coefficient in 1/s as a scalar or array.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L674-L696)

### particula.dynamics.wall_loss.ChargedWallLossStrategy.loss_coefficient_for_particles

*method*

```python
def loss_coefficient_for_particles(particle_radius: NDArray[np.float64], particle_density: NDArray[np.float64], temperature: float, pressure: float) -> NDArray[np.float64]
```

Compute charged wall loss coefficient for provided particle arrays.

This charge-free interface assumes neutral particles. For charged
material, use ``compute_coefficient_from_arrays`` with explicit charge
arrays. No earlier particle evaluation supplies implicit charge state.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Particle radii in meters.
- `particle_density` (`NDArray[np.float64]`) — Particle densities in kg/m³.
- `temperature` (`float`) — Gas temperature in kelvin.
- `pressure` (`float`) — Gas pressure in pascals.

**Returns**

- (`NDArray[np.float64]`) — Wall loss coefficient in 1/s for each particle.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L698-L742)

### particula.dynamics.wall_loss.ChargedWallLossStrategy.step

*method*

```python
def step(particle: ParticleData, temperature: float, pressure: float, time_step: float) -> ParticleData
```

Advance particle-resolved concentration with charged wall loss.

Computes survival probabilities using neutral, image-charge, and
optional drift contributions. Clamps probabilities to [0, 1] and
updates particle concentrations and distributions.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to update.
- `temperature` (`float`) — Gas temperature in kelvin.
- `pressure` (`float`) — Gas pressure in pascals.
- `time_step` (`float`) — Time step in seconds.

**Returns**

- (`ParticleData`) — Updated native particle data.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L790-L812)

### particula.dynamics.wall_loss.ChargedWallLossStrategy.loss_rate

*method*

```python
def loss_rate(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Return the wall loss rate for the given state.

The loss rate is computed as ``-k(Dp) * c`` where ``k(Dp)`` is the
size-dependent wall loss coefficient and ``c`` is the particle number
concentration.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to evaluate.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Wall loss rate [#/m^3 s] (negative values indicate loss).

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L220-L240)

### particula.dynamics.wall_loss.ChargedWallLossStrategy.rate

*method*

```python
def rate(particle: ParticleData, temperature: float, pressure: float) -> NDArray[np.float64]
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Return the wall loss rate as an array.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to evaluate.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`NDArray[np.float64]`) — Array of wall loss rates [#/m^3 s].

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L242-L264)

## particula.dynamics.wall_loss.RectangularWallLossBuilder

*class*

```python
class RectangularWallLossBuilder(BuilderABC, BuilderWallEddyDiffusivityMixin, BuilderChamberDimensionsMixin, BuilderDistributionTypeMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderWallEddyDiffusivityMixin`, `particula.builder_mixin.BuilderChamberDimensionsMixin`, `particula.builder_mixin.BuilderDistributionTypeMixin`

Re-exported from: `particula.dynamics.wall_loss.wall_loss_builders`

Builder for rectangular wall loss strategies.

Validates wall eddy diffusivity, chamber dimensions, and distribution type
before constructing :class:`RectangularWallLossStrategy`.

**Attributes**

- `wall_eddy_diffusivity` (`Optional[float]`) — Wall eddy diffusivity in 1/s.
- `chamber_dimensions` (`Optional[Tuple[float, float, float]]`) — Tuple of (length, width, height) in meters.
- `distribution_type` (`str`) — Distribution type for the strategy output.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_builders.py#L74-L123)

### particula.dynamics.wall_loss.RectangularWallLossBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.wall_loss.RectangularWallLossBuilder.wall_eddy_diffusivity

*attribute* · *instance attribute*

```python
wall_eddy_diffusivity: Optional[float] = None
```

Inherited from: `particula.builder_mixin.BuilderWallEddyDiffusivityMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L616-L616)

### particula.dynamics.wall_loss.RectangularWallLossBuilder.chamber_dimensions

*attribute* · *instance attribute*

```python
chamber_dimensions: Optional[Tuple[float, float, float]] = None
```

Inherited from: `particula.builder_mixin.BuilderChamberDimensionsMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L712-L712)

### particula.dynamics.wall_loss.RectangularWallLossBuilder.distribution_type

*attribute* · *instance attribute*

```python
distribution_type: str = 'discrete'
```

Inherited from: `particula.builder_mixin.BuilderDistributionTypeMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L944-L944)

### particula.dynamics.wall_loss.RectangularWallLossBuilder.__init__

*method*

```python
def __init__()
```

Initialize the rectangular wall loss builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_builders.py#L91-L102)

### particula.dynamics.wall_loss.RectangularWallLossBuilder.build

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

### particula.dynamics.wall_loss.RectangularWallLossBuilder.check_keys

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

### particula.dynamics.wall_loss.RectangularWallLossBuilder.pre_build_check

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

### particula.dynamics.wall_loss.RectangularWallLossBuilder.set_parameters

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

### particula.dynamics.wall_loss.RectangularWallLossBuilder.set_wall_eddy_diffusivity

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

### particula.dynamics.wall_loss.RectangularWallLossBuilder.set_chamber_dimensions

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

### particula.dynamics.wall_loss.RectangularWallLossBuilder.set_distribution_type

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

## particula.dynamics.wall_loss.RectangularWallLossStrategy

*class*

```python
class RectangularWallLossStrategy(WallLossStrategy)
```

Bases: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Re-exported from: `particula.dynamics.wall_loss.wall_loss_strategies`

Wall loss strategy for rectangular (box) chambers.

Calculates particle wall deposition in rectangular chamber geometry
using turbulent diffusion and gravitational settling. Supports
discrete, continuous PDF, and particle-resolved distributions.

**Attributes**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity [1/s].
- `chamber_dimensions` (`Tuple[float, float, float]`) — Chamber dimensions (length, width, height) [m].
- `distribution_type` (`str`) — Distribution type ("discrete", "continuous_pdf", or "particle_resolved").

**Examples**

```pycon
>>> import particula as par
>>> particle = par.particles.create_lognormal_binned_particle_data(
...     distribution_type="discrete",
... )
>>> strategy = par.dynamics.RectangularWallLossStrategy(
...     wall_eddy_diffusivity=0.001,
...     chamber_dimensions=(1.0, 0.5, 0.5),
...     distribution_type="discrete",
... )
>>> rate = strategy.rate(
...     particle=particle,
...     temperature=298.0,
...     pressure=101325.0,
... )
>>> _ = strategy.step(
...     particle=particle,
...     temperature=298.0,
...     pressure=101325.0,
...     time_step=0.1,
... )
```

> **References**
>
> Crump, J. G., & Seinfeld, J. H. (1981). Turbulent deposition and
> gravitational sedimentation of an aerosol in a vessel of arbitrary
> shape. Journal of Aerosol Science, 12(5), 405–415.
> McMurry, P. H., & Rader, D. J. (1985). Aerosol wall losses in
> electrically charged chambers. Aerosol Science and Technology, 4(3),
> 249–268.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L815-L946)

### particula.dynamics.wall_loss.RectangularWallLossStrategy.chamber_dimensions

*attribute* · *instance attribute*

```python
chamber_dimensions: Tuple[float, float, float] = (float(chamber_dimensions[0]), float(chamber_dimensions[1]), float(chamber_dimensions[2]))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L892-L896)

### particula.dynamics.wall_loss.RectangularWallLossStrategy.distribution_type

*attribute* · *instance attribute*

```python
distribution_type: str = distribution_type
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L176-L176)

### particula.dynamics.wall_loss.RectangularWallLossStrategy.random_generator

*attribute* · *instance attribute*

```python
random_generator: np.random.Generator = np.random.default_rng()
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L177-L177)

### particula.dynamics.wall_loss.RectangularWallLossStrategy.wall_eddy_diffusivity

*attribute* · *instance attribute*

```python
wall_eddy_diffusivity: float = wall_eddy_diffusivity
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L175-L175)

### particula.dynamics.wall_loss.RectangularWallLossStrategy.__init__

*method*

```python
def __init__(wall_eddy_diffusivity: float, chamber_dimensions: Tuple[float, float, float], distribution_type: str = 'discrete') -> None
```

Initialize rectangular wall loss strategy.

**Parameters**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity [1/s].
- `chamber_dimensions` (`Tuple[float, float, float]`) — Chamber dimensions (length, width, height) in meters. All must be positive.
- `distribution_type` (`str`) (default: `'discrete'`) — Distribution type ("discrete", "continuous_pdf", or "particle_resolved").

**Raises**

- `ValueError` — If ``chamber_dimensions`` does not contain exactly
three positive values.
- `ValueError` — If ``distribution_type`` is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L861-L896)

### particula.dynamics.wall_loss.RectangularWallLossStrategy.loss_coefficient

*method*

```python
def loss_coefficient(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Return the rectangular wall loss coefficient for the state.

**Parameters**

- `particle` (`ParticleData`) — Native particle data providing radius and density.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Wall loss coefficient [1/s] for each particle bin.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L898-L914)

### particula.dynamics.wall_loss.RectangularWallLossStrategy.loss_coefficient_for_particles

*method*

```python
def loss_coefficient_for_particles(particle_radius: NDArray[np.float64], particle_density: NDArray[np.float64], temperature: float, pressure: float) -> NDArray[np.float64]
```

Return wall loss coefficient for provided particle properties.

This method is used for particle-resolved simulations to evaluate
coefficients on active particles without reconstructing the full
representation.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Particle radii [m].
- `particle_density` (`NDArray[np.float64]`) — Particle densities [kg/m^3].
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`NDArray[np.float64]`) — Wall loss coefficient [1/s] for each particle.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L916-L946)

### particula.dynamics.wall_loss.RectangularWallLossStrategy.loss_rate

*method*

```python
def loss_rate(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Return the wall loss rate for the given state.

The loss rate is computed as ``-k(Dp) * c`` where ``k(Dp)`` is the
size-dependent wall loss coefficient and ``c`` is the particle number
concentration.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to evaluate.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Wall loss rate [#/m^3 s] (negative values indicate loss).

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L220-L240)

### particula.dynamics.wall_loss.RectangularWallLossStrategy.rate

*method*

```python
def rate(particle: ParticleData, temperature: float, pressure: float) -> NDArray[np.float64]
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Return the wall loss rate as an array.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to evaluate.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`NDArray[np.float64]`) — Array of wall loss rates [#/m^3 s].

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L242-L264)

### particula.dynamics.wall_loss.RectangularWallLossStrategy.step

*method*

```python
def step(particle: ParticleData, temperature: float, pressure: float, time_step: float) -> ParticleData
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Advance the native particle data by one wall loss step.

For ``"discrete"`` and ``"continuous_pdf"`` distributions, this
applies a deterministic first-order loss to the number
concentration in each bin. For ``"particle_resolved"``, it
applies a stochastic removal process using binomial random draws
based on survival probability.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to update.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].
- `time_step` (`float`) — Time step [s].

**Returns**

- (`ParticleData`) — Updated native particle data (same instance).

**Raises**

- `ValueError` — If ``distribution_type`` is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L266-L293)

## particula.dynamics.wall_loss.SphericalWallLossBuilder

*class*

```python
class SphericalWallLossBuilder(BuilderABC, BuilderWallEddyDiffusivityMixin, BuilderChamberRadiusMixin, BuilderDistributionTypeMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderWallEddyDiffusivityMixin`, `particula.builder_mixin.BuilderChamberRadiusMixin`, `particula.builder_mixin.BuilderDistributionTypeMixin`

Re-exported from: `particula.dynamics.wall_loss.wall_loss_builders`

Builder for spherical wall loss strategies.

Validates wall eddy diffusivity, chamber radius, and distribution type
before constructing :class:`SphericalWallLossStrategy`.

**Attributes**

- `wall_eddy_diffusivity` (`Optional[float]`) — Wall eddy diffusivity in 1/s.
- `chamber_radius` (`Optional[float]`) — Chamber radius in meters.
- `distribution_type` (`str`) — Distribution type for the strategy output.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_builders.py#L25-L71)

### particula.dynamics.wall_loss.SphericalWallLossBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.wall_loss.SphericalWallLossBuilder.wall_eddy_diffusivity

*attribute* · *instance attribute*

```python
wall_eddy_diffusivity: Optional[float] = None
```

Inherited from: `particula.builder_mixin.BuilderWallEddyDiffusivityMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L616-L616)

### particula.dynamics.wall_loss.SphericalWallLossBuilder.chamber_radius

*attribute* · *instance attribute*

```python
chamber_radius: Optional[float] = None
```

Inherited from: `particula.builder_mixin.BuilderChamberRadiusMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L663-L663)

### particula.dynamics.wall_loss.SphericalWallLossBuilder.distribution_type

*attribute* · *instance attribute*

```python
distribution_type: str = 'discrete'
```

Inherited from: `particula.builder_mixin.BuilderDistributionTypeMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L944-L944)

### particula.dynamics.wall_loss.SphericalWallLossBuilder.__init__

*method*

```python
def __init__()
```

Initialize the spherical wall loss builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_builders.py#L42-L50)

### particula.dynamics.wall_loss.SphericalWallLossBuilder.build

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

### particula.dynamics.wall_loss.SphericalWallLossBuilder.check_keys

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

### particula.dynamics.wall_loss.SphericalWallLossBuilder.pre_build_check

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

### particula.dynamics.wall_loss.SphericalWallLossBuilder.set_parameters

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

### particula.dynamics.wall_loss.SphericalWallLossBuilder.set_wall_eddy_diffusivity

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

### particula.dynamics.wall_loss.SphericalWallLossBuilder.set_chamber_radius

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

### particula.dynamics.wall_loss.SphericalWallLossBuilder.set_distribution_type

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

## particula.dynamics.wall_loss.SphericalWallLossStrategy

*class*

```python
class SphericalWallLossStrategy(WallLossStrategy)
```

Bases: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Re-exported from: `particula.dynamics.wall_loss.wall_loss_strategies`

Wall loss strategy for spherical chambers.

Calculates particle wall deposition in spherical chamber geometry
using turbulent diffusion and gravitational settling.

**Attributes**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity [1/s].
- `chamber_radius` (`float`) — Radius of the spherical chamber [m].
- `distribution_type` (`str`) — Distribution type ("discrete", "continuous_pdf", or "particle_resolved").

**Examples**

```pycon
>>> from particula.dynamics.wall_loss.wall_loss_strategies import (
...     SphericalWallLossStrategy,
... )
>>> strategy = SphericalWallLossStrategy(
...     wall_eddy_diffusivity=0.001,
...     chamber_radius=0.5,
...     distribution_type="discrete",
... )
>>> rate = strategy.rate(
...     particle=particle,
...     temperature=298.0,
...     pressure=101325.0,
... )
>>> particle = strategy.step(
...     particle=particle,
...     temperature=298.0,
...     pressure=101325.0,
...     time_step=1.0,
... )
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L296-L408)

### particula.dynamics.wall_loss.SphericalWallLossStrategy.chamber_radius

*attribute* · *instance attribute*

```python
chamber_radius: float = chamber_radius
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L356-L356)

### particula.dynamics.wall_loss.SphericalWallLossStrategy.distribution_type

*attribute* · *instance attribute*

```python
distribution_type: str = distribution_type
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L176-L176)

### particula.dynamics.wall_loss.SphericalWallLossStrategy.random_generator

*attribute* · *instance attribute*

```python
random_generator: np.random.Generator = np.random.default_rng()
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L177-L177)

### particula.dynamics.wall_loss.SphericalWallLossStrategy.wall_eddy_diffusivity

*attribute* · *instance attribute*

```python
wall_eddy_diffusivity: float = wall_eddy_diffusivity
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L175-L175)

### particula.dynamics.wall_loss.SphericalWallLossStrategy.__init__

*method*

```python
def __init__(wall_eddy_diffusivity: float, chamber_radius: float, distribution_type: str = 'discrete') -> None
```

Initialize spherical wall loss strategy.

**Parameters**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity [1/s].
- `chamber_radius` (`float`) — Radius of the spherical chamber [m].
- `distribution_type` (`str`) (default: `'discrete'`) — Distribution type ("discrete", "continuous_pdf", or "particle_resolved").

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L332-L356)

### particula.dynamics.wall_loss.SphericalWallLossStrategy.loss_coefficient

*method*

```python
def loss_coefficient(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Return the spherical wall loss coefficient for the given state.

The coefficient is computed from the system state using
:func:`get_spherical_wall_loss_coefficient_via_system_state`.

**Parameters**

- `particle` (`ParticleData`) — Native particle data providing radius and density.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Wall loss coefficient [1/s].

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L358-L377)

### particula.dynamics.wall_loss.SphericalWallLossStrategy.loss_coefficient_for_particles

*method*

```python
def loss_coefficient_for_particles(particle_radius: NDArray[np.float64], particle_density: NDArray[np.float64], temperature: float, pressure: float) -> NDArray[np.float64]
```

Return the wall loss coefficient for given particle properties.

This method is used for particle-resolved simulations where we
need to calculate coefficients for a subset of active particles.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Particle radii [m].
- `particle_density` (`NDArray[np.float64]`) — Particle densities [kg/m^3].
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`NDArray[np.float64]`) — Wall loss coefficient [1/s] for each particle.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L379-L408)

### particula.dynamics.wall_loss.SphericalWallLossStrategy.loss_rate

*method*

```python
def loss_rate(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Return the wall loss rate for the given state.

The loss rate is computed as ``-k(Dp) * c`` where ``k(Dp)`` is the
size-dependent wall loss coefficient and ``c`` is the particle number
concentration.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to evaluate.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Wall loss rate [#/m^3 s] (negative values indicate loss).

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L220-L240)

### particula.dynamics.wall_loss.SphericalWallLossStrategy.rate

*method*

```python
def rate(particle: ParticleData, temperature: float, pressure: float) -> NDArray[np.float64]
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Return the wall loss rate as an array.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to evaluate.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`NDArray[np.float64]`) — Array of wall loss rates [#/m^3 s].

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L242-L264)

### particula.dynamics.wall_loss.SphericalWallLossStrategy.step

*method*

```python
def step(particle: ParticleData, temperature: float, pressure: float, time_step: float) -> ParticleData
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Advance the native particle data by one wall loss step.

For ``"discrete"`` and ``"continuous_pdf"`` distributions, this
applies a deterministic first-order loss to the number
concentration in each bin. For ``"particle_resolved"``, it
applies a stochastic removal process using binomial random draws
based on survival probability.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to update.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].
- `time_step` (`float`) — Time step [s].

**Returns**

- (`ParticleData`) — Updated native particle data (same instance).

**Raises**

- `ValueError` — If ``distribution_type`` is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L266-L293)

## particula.dynamics.wall_loss.WallLossFactory

*class*

```python
class WallLossFactory(StrategyFactoryABC)
```

Bases: `particula.abc_factory.StrategyFactoryABC`

Re-exported from: `particula.dynamics.wall_loss.wall_loss_factories`

Factory for wall loss strategies.

Builds wall loss strategies using registered builders and the
:class:`StrategyFactoryABC` workflow.

> **Supported strategy types**
>
> - "spherical": ``SphericalWallLossStrategy`` for spherical chambers.
> - "rectangular": ``RectangularWallLossStrategy`` for box chambers.
> - "charged": ``ChargedWallLossStrategy`` with image-charge
>   enhancement and optional electric-field drift; requires geometry
>   and matching size parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_factories.py#L23-L82)

### particula.dynamics.wall_loss.WallLossFactory.get_builders

*method*

```python
def get_builders() -> Dict[str, BuilderType]
```

Return available wall loss builders keyed by strategy name.

**Returns**

- (`Dict[str, BuilderType]`) — Dict[str, BuilderType]: Mapping of strategy type to builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_factories.py#L37-L47)

### particula.dynamics.wall_loss.WallLossFactory.get_strategy

*method*

```python
def get_strategy(strategy_type: str, parameters: Optional[Dict[str, Any]] = None)
```

Create a wall loss strategy using its corresponding builder.

**Parameters**

- `strategy_type` (`str`) — Strategy name ("spherical", "rectangular", or "charged").
- `parameters` (`Optional[Dict[str, Any]]`) (default: `None`) — Optional parameter mapping for the builder. Include geometry keys (``chamber_radius`` or ``chamber_dimensions``) plus optional ``wall_potential`` and ``wall_electric_field`` entries, and ``distribution_type`` when needed.

**Returns**

- `result` (`WallLossStrategy`) — Built wall loss strategy instance.

**Raises**

- `ValueError` — If ``strategy_type`` is unknown or parameters are
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_factories.py#L49-L82)

## particula.dynamics.wall_loss.WallLossStrategy

*class*

```python
class WallLossStrategy(ABC)
```

Bases: `ABC`

Re-exported from: `particula.dynamics.wall_loss.wall_loss_strategies`

Abstract base class for wall loss strategies.

Wall loss strategies compute wall loss coefficients and associated
rates for different distribution types.

**Attributes**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity [1/s].
- `distribution_type` (`str`) — Distribution type ("discrete", "continuous_pdf", or "particle_resolved").
- `random_generator` (`np.random.Generator`) — Random number generator for stochastic particle-resolved simulations.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L133-L293)

### particula.dynamics.wall_loss.WallLossStrategy.distribution_type

*attribute* · *instance attribute*

```python
distribution_type: str = distribution_type
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L176-L176)

### particula.dynamics.wall_loss.WallLossStrategy.random_generator

*attribute* · *instance attribute*

```python
random_generator: np.random.Generator = np.random.default_rng()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L177-L177)

### particula.dynamics.wall_loss.WallLossStrategy.wall_eddy_diffusivity

*attribute* · *instance attribute*

```python
wall_eddy_diffusivity: float = wall_eddy_diffusivity
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L175-L175)

### particula.dynamics.wall_loss.WallLossStrategy.__init__

*method*

```python
def __init__(wall_eddy_diffusivity: float, distribution_type: str) -> None
```

Initialize the wall loss strategy.

**Parameters**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity [1/s].
- `distribution_type` (`str`) — Distribution type ("discrete", "continuous_pdf", or "particle_resolved").

**Raises**

- `ValueError` — If ``distribution_type`` is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L151-L177)

### particula.dynamics.wall_loss.WallLossStrategy.loss_coefficient

*method* · *abstract*

```python
def loss_coefficient(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Return the wall loss coefficient for the given state.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to evaluate.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Wall loss coefficient [1/s].

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L179-L195)

### particula.dynamics.wall_loss.WallLossStrategy.loss_coefficient_for_particles

*method* · *abstract*

```python
def loss_coefficient_for_particles(particle_radius: NDArray[np.float64], particle_density: NDArray[np.float64], temperature: float, pressure: float) -> NDArray[np.float64]
```

Return the wall loss coefficient for given particle properties.

This method is used for particle-resolved simulations where we
need to calculate coefficients for a subset of active particles.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Particle radii [m].
- `particle_density` (`NDArray[np.float64]`) — Particle densities [kg/m^3].
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`NDArray[np.float64]`) — Wall loss coefficient [1/s] for each particle.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L197-L218)

### particula.dynamics.wall_loss.WallLossStrategy.loss_rate

*method*

```python
def loss_rate(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Return the wall loss rate for the given state.

The loss rate is computed as ``-k(Dp) * c`` where ``k(Dp)`` is the
size-dependent wall loss coefficient and ``c`` is the particle number
concentration.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to evaluate.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Wall loss rate [#/m^3 s] (negative values indicate loss).

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L220-L240)

### particula.dynamics.wall_loss.WallLossStrategy.rate

*method*

```python
def rate(particle: ParticleData, temperature: float, pressure: float) -> NDArray[np.float64]
```

Return the wall loss rate as an array.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to evaluate.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`NDArray[np.float64]`) — Array of wall loss rates [#/m^3 s].

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L242-L264)

### particula.dynamics.wall_loss.WallLossStrategy.step

*method*

```python
def step(particle: ParticleData, temperature: float, pressure: float, time_step: float) -> ParticleData
```

Advance the native particle data by one wall loss step.

For ``"discrete"`` and ``"continuous_pdf"`` distributions, this
applies a deterministic first-order loss to the number
concentration in each bin. For ``"particle_resolved"``, it
applies a stochastic removal process using binomial random draws
based on survival probability.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to update.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].
- `time_step` (`float`) — Time step [s].

**Returns**

- (`ParticleData`) — Updated native particle data (same instance).

**Raises**

- `ValueError` — If ``distribution_type`` is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L266-L293)

## particula.dynamics.wall_loss.get_charged_wall_loss_rate

*function*

```python
def get_charged_wall_loss_rate(wall_eddy_diffusivity: float, particle_radius: Union[float, NDArray[np.float64]], particle_density: Union[float, NDArray[np.float64]], particle_concentration: Union[float, NDArray[np.float64]], particle_charge: Union[float, NDArray[np.float64]], temperature: float, pressure: float, chamber_geometry: str, chamber_radius: Union[float, None] = None, chamber_dimensions: Union[Tuple[float, float, float], None] = None, wall_potential: float = 0.0, wall_electric_field: Union[float, Tuple[float, float, float]] = 0.0) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.dynamics.wall_loss.rate`

Calculate charged wall loss rate for spherical or rectangular chambers.

Combines neutral deposition with image-charge enhancement and optional
electric-field drift. Image-charge effects apply when particles carry
charge even if ``wall_potential`` is zero; drift is applied only when
``wall_electric_field`` is non-zero. Reduces to the neutral rate when
both charge and electric field are zero.

**Parameters**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity in s⁻¹.
- `particle_radius` (`Union[float, NDArray[np.float64]]`) — Particle radius in meters.
- `particle_density` (`Union[float, NDArray[np.float64]]`) — Particle density in kg/m³.
- `particle_concentration` (`Union[float, NDArray[np.float64]]`) — Particle concentration in #/m³.
- `particle_charge` (`Union[float, NDArray[np.float64]]`) — Particle charge in elementary charge units.
- `temperature` (`float`) — Gas temperature in kelvin.
- `pressure` (`float`) — Gas pressure in pascals.
- `chamber_geometry` (`str`) — Geometry name, ``"spherical"`` or ``"rectangular"``.
- `chamber_radius` (`Union[float, None]`) (default: `None`) — Chamber radius in meters for spherical geometry.
- `chamber_dimensions` (`Union[Tuple[float, float, float], None]`) (default: `None`) — ``(length, width, height)`` in meters for rectangular geometry.
- `wall_potential` (`float`) (default: `0.0`) — Wall potential in volts. Zero still enables image-charge effects when charge is non-zero.
- `wall_electric_field` (`Union[float, Tuple[float, float, float]]`) (default: `0.0`) — Electric field magnitude in V/m (scalar for spherical or tuple for rectangular). Zero disables drift.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Charged wall loss rate in #/m³·s as scalar or array matching
- (`Union[float, NDArray[np.float64]]`) — ``particle_concentration``.

**Raises**

- `ValueError` — If geometry arguments are inconsistent or not finite.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/rate.py#L142-L208)

## particula.dynamics.wall_loss.get_particle_resolved_wall_loss_step

*function*

```python
def get_particle_resolved_wall_loss_step(particle_radius: NDArray[np.float64], particle_density: NDArray[np.float64], concentration: NDArray[np.float64], loss_coefficient_func: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]], time_step: float, random_generator: np.random.Generator) -> NDArray[np.float64]
```

Re-exported from: `particula.dynamics.wall_loss.wall_loss_strategies`

Perform particle-resolved wall loss step with stochastic removal.

For particle-resolved simulations, each computational particle survives
with probability ``exp(-k * dt)``, where ``k`` is the size-dependent wall
loss coefficient. Binomial draws determine which particles are lost to the
walls.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Array of particle radii [m].
- `particle_density` (`NDArray[np.float64]`) — Array of particle densities [kg/m^3].
- `concentration` (`NDArray[np.float64]`) — Array of particle concentrations [#/m^3].
- `loss_coefficient_func` (`Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]`) — Callable returning wall loss coefficient for provided particle radius and density. Temperature, pressure, and geometry should be bound via closure.
- `time_step` (`float`) — Time step [s].
- `random_generator` (`np.random.Generator`) — Random number generator for stochastic draws.

**Returns**

- (`NDArray[np.float64]`) — Array of survival indicators (1.0 for survived, 0.0 for lost).

**Examples**

```pycon
>>> import numpy as np
>>> from particula.dynamics.wall_loss.wall_loss_strategies import (
...     get_particle_resolved_wall_loss_step,
... )
>>> radius = np.array([1e-7, 2e-7, 3e-7])
>>> density = np.array([1000.0, 1000.0, 1000.0])
>>> concentration = np.array([1e6, 1e6, 1e6])
>>> def coeff_func(r, d):
...     return 1e-4 * np.ones_like(r)
>>> rng = np.random.default_rng(42)
>>> survived = get_particle_resolved_wall_loss_step(
...     particle_radius=radius,
...     particle_density=density,
...     concentration=concentration,
...     loss_coefficient_func=coeff_func,
...     time_step=1.0,
...     random_generator=rng,
... )
>>> print(survived)
[1. 1. 1.]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L52-L130)

## particula.dynamics.wall_loss.get_rectangle_wall_loss_rate

*function*

```python
def get_rectangle_wall_loss_rate(wall_eddy_diffusivity: float, particle_radius: Union[float, NDArray[np.float64]], particle_density: Union[float, NDArray[np.float64]], particle_concentration: Union[float, NDArray[np.float64]], temperature: float, pressure: float, chamber_dimensions: Tuple[float, float, float]) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.dynamics.wall_loss.rate`

Calculate the wall loss rate for a rectangular chamber.

The loss rate is computed as ``L = -k c`` where ``k`` is the wall
loss coefficient [1/s] and ``c`` is the particle concentration
[#/m³]. The coefficient is obtained from the rectangular wall loss
coefficient computed via
:func:`get_rectangle_wall_loss_coefficient_via_system_state`.

**Parameters**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity [s⁻¹].
- `particle_radius` (`Union[float, NDArray[np.float64]]`) — Particle radius [m].
- `particle_density` (`Union[float, NDArray[np.float64]]`) — Particle density [kg/m³].
- `particle_concentration` (`Union[float, NDArray[np.float64]]`) — Particle concentration [#/m³].
- `temperature` (`float`) — System temperature [K].
- `pressure` (`float`) — System pressure [Pa].
- `chamber_dimensions` (`Tuple[float, float, float]`) — Chamber dimensions ``(length, width, height)`` [m].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Wall loss rate [#/m³·s] as a scalar or array.

**Examples**

```pycon
>>> import particula as par
>>> loss_rate = par.dynamics.wall_loss.get_rectangle_wall_loss_rate(
...     wall_eddy_diffusivity=1e-4,
...     particle_radius=5e-8,
...     particle_density=1200.0,
...     particle_concentration=2e10,
...     temperature=300.0,
...     pressure=101325.0,
...     chamber_dimensions=(1.0, 0.5, 0.5),
... )
>>> loss_rate
```

> **References**
>
> Hinds, W. C. "Aerosol Technology." 2nd ed. John Wiley & Sons, 1999.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/rate.py#L83-L139)

## particula.dynamics.wall_loss.get_spherical_wall_loss_rate

*function*

```python
def get_spherical_wall_loss_rate(wall_eddy_diffusivity: float, particle_radius: Union[float, NDArray[np.float64]], particle_density: Union[float, NDArray[np.float64]], particle_concentration: Union[float, NDArray[np.float64]], temperature: float, pressure: float, chamber_radius: float) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.dynamics.wall_loss.rate`

Calculate the wall loss rate for a spherical chamber.

The loss rate is computed as ``L = -k c`` where ``k`` is the wall
loss coefficient [1/s] and ``c`` is the particle concentration
[#/m³]. The coefficient is obtained from the spherical wall loss
coefficient computed via
:func:`get_spherical_wall_loss_coefficient_via_system_state`.

**Parameters**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity [s⁻¹].
- `particle_radius` (`Union[float, NDArray[np.float64]]`) — Particle radius [m].
- `particle_density` (`Union[float, NDArray[np.float64]]`) — Particle density [kg/m³].
- `particle_concentration` (`Union[float, NDArray[np.float64]]`) — Particle concentration [#/m³].
- `temperature` (`float`) — System temperature [K].
- `pressure` (`float`) — System pressure [Pa].
- `chamber_radius` (`float`) — Radius of the spherical chamber [m].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Wall loss rate [#/m³·s] as a scalar or array.

**Examples**

```pycon
>>> import particula as par
>>> rate = par.dynamics.wall_loss.get_spherical_wall_loss_rate(
...     wall_eddy_diffusivity=1e-3,
...     particle_radius=1e-7,
...     particle_density=1000.0,
...     particle_concentration=1e11,
...     temperature=298.0,
...     pressure=101325.0,
...     chamber_radius=0.5,
... )
>>> rate
```

> **References**
>
> Wikipedia contributors. "Aerosol dynamics." *Wikipedia*.
> https://en.wikipedia.org/wiki/Aerosol.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/rate.py#L22-L79)

**Modules**

- `particula.dynamics.wall_loss._native`
- `particula.dynamics.wall_loss.rate`
- `particula.dynamics.wall_loss.wall_loss_builders`
- `particula.dynamics.wall_loss.wall_loss_factories`
- `particula.dynamics.wall_loss.wall_loss_strategies`
