# particula.builder_mixin

*module*

Builder mixin helpers for assigning particle attributes and units.

This module defines mixins that provide builders with fluent setters for
physical quantities such as density, surface tension, concentration, and
chamber geometry while handling unit conversions and input validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L1-L1)

## particula.builder_mixin.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L19-L19)

## particula.builder_mixin.BuilderChamberDimensionsMixin

*class*

```python
class BuilderChamberDimensionsMixin
```

Mixin for setting rectangular chamber dimensions.

Adds ``set_chamber_dimensions`` for builders that require rectangular
chamber geometry and stores dimensions in meters.

**Attributes**

- `chamber_dimensions` (`Optional[Tuple[float, float, float]]`) — Tuple of (length, width, height) in meters.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L700-L779)

### particula.builder_mixin.BuilderChamberDimensionsMixin.chamber_dimensions

*attribute* · *instance attribute*

```python
chamber_dimensions: Optional[Tuple[float, float, float]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L712-L712)

### particula.builder_mixin.BuilderChamberDimensionsMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize chamber dimensions mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L710-L712)

### particula.builder_mixin.BuilderChamberDimensionsMixin.set_chamber_dimensions

*method*

```python
def set_chamber_dimensions(chamber_dimensions: Tuple[float, float, float], chamber_dimensions_units: str = 'm') -> BuilderChamberDimensionsMixin
```

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

## particula.builder_mixin.BuilderChamberRadiusMixin

*class*

```python
class BuilderChamberRadiusMixin
```

Mixin for setting spherical chamber radius.

Adds ``set_chamber_radius`` for builders that require spherical chamber
geometry and stores the value in meters.

**Attributes**

- `chamber_radius` (`Optional[float]`) — Chamber radius in meters.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L651-L697)

### particula.builder_mixin.BuilderChamberRadiusMixin.chamber_radius

*attribute* · *instance attribute*

```python
chamber_radius: Optional[float] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L663-L663)

### particula.builder_mixin.BuilderChamberRadiusMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize chamber radius mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L661-L663)

### particula.builder_mixin.BuilderChamberRadiusMixin.set_chamber_radius

*method*

```python
def set_chamber_radius(chamber_radius: float, chamber_radius_units: str = 'm') -> BuilderChamberRadiusMixin
```

Set the chamber radius in meters.

**Parameters**

- `chamber_radius` (`float`) — Chamber radius value.
- `chamber_radius_units` (`str`) (default: `'m'`) — Units for the value. Defaults to "m".

**Returns**

- `self` (`BuilderChamberRadiusMixin`) — Self for method chaining.

**Raises**

- `ValueError` — If ``chamber_radius`` is not positive.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L665-L697)

## particula.builder_mixin.BuilderChargeMixin

*class*

```python
class BuilderChargeMixin
```

Mixin that stores a particle charge count.

**Attributes**

- `charge` (`Optional[Union[float, NDArray[np.float64]]]`) — Assigned charge in elemental charge units (dimensionless).

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L217-L245)

### particula.builder_mixin.BuilderChargeMixin.charge

*attribute* · *instance attribute*

```python
charge: Optional[Union[float, NDArray[np.float64]]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L226-L226)

### particula.builder_mixin.BuilderChargeMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize charge mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L224-L226)

### particula.builder_mixin.BuilderChargeMixin.set_charge

*method*

```python
def set_charge(charge: Union[float, NDArray[np.float64]], charge_units: Optional[str] = None) -> Self
```

Set the particle charge count.

**Parameters**

- `charge` (`Union[float, NDArray[np.float64]]`) — Numeric value describing elemental charges.
- `charge_units` (`Optional[str]`) (default: `None`) — Optional units that are logged and ignored.

**Returns**

- `self` (`BuilderChargeMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L228-L245)

## particula.builder_mixin.BuilderConcentrationMixin

*class*

```python
class BuilderConcentrationMixin
```

Mixin that stores concentration values in default units.

The mixin exposes `set_concentration` so builders can accept concentration
inputs in arbitrary units and persist them in the configured default units.

**Attributes**

- `concentration` (`Optional[Union[float, NDArray[np.float64]]]`) — Stored concentration in `default_units`.
- `default_units` — Default units applied when no conversion is needed.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L176-L214)

### particula.builder_mixin.BuilderConcentrationMixin.concentration

*attribute* · *instance attribute*

```python
concentration: Optional[Union[float, NDArray[np.float64]]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L189-L189)

### particula.builder_mixin.BuilderConcentrationMixin.default_units

*attribute* · *instance attribute*

```python
default_units = default_units if default_units else 'kg/m^3'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L190-L190)

### particula.builder_mixin.BuilderConcentrationMixin.__init__

*method*

```python
def __init__(default_units: str = 'kg/m^3') -> None
```

Initialize concentration mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L187-L190)

### particula.builder_mixin.BuilderConcentrationMixin.set_concentration

*method*

```python
def set_concentration(concentration: Union[float, NDArray[np.float64]], concentration_units: str) -> Self
```

Assign the concentration normalized to the default units.

**Parameters**

- `concentration` (`Union[float, NDArray[np.float64]]`) — Non-negative concentration value to store.
- `concentration_units` (`str`) — Units for ``concentration``. Converted to ``self.default_units`` before storage.

**Returns**

- `self` (`BuilderConcentrationMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L192-L214)

## particula.builder_mixin.BuilderDensityMixin

*class*

```python
class BuilderDensityMixin
```

Mixin that stores a particle density with unit conversion.

This mixin provides `set_density` so builders can accept density values
in arbitrary units and normalize them to kilograms per cubic meter.

**Attributes**

- `density` (`Optional[Union[float, NDArray[np.float64]]]`) — Particle density in kg/m^3 after conversion.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L22-L56)

### particula.builder_mixin.BuilderDensityMixin.density

*attribute* · *instance attribute*

```python
density: Optional[Union[float, NDArray[np.float64]]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L34-L34)

### particula.builder_mixin.BuilderDensityMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize density mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L32-L34)

### particula.builder_mixin.BuilderDensityMixin.set_density

*method*

```python
def set_density(density: Union[float, NDArray[np.float64]], density_units: str)
```

Set the particle density in kilograms per cubic meter.

**Parameters**

- `density` (`Union[float, NDArray[np.float64]]`) — Positive density value to store.
- `density_units` (`str`) — Units for ``density``. Defaults to ``"kg/m^3"`` and accepts any unit supported by the converter.

**Returns**

- `self` (`BuilderDensityMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L36-L56)

## particula.builder_mixin.BuilderDistributionTypeMixin

*class*

```python
class BuilderDistributionTypeMixin
```

Mixin for setting distribution type.

Adds ``set_distribution_type`` for builders requiring a distribution type.
Defaults to "discrete" and validates against supported types.

**Attributes**

- `distribution_type` (`str`) — Distribution type string.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L932-L966)

### particula.builder_mixin.BuilderDistributionTypeMixin.distribution_type

*attribute* · *instance attribute*

```python
distribution_type: str = 'discrete'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L944-L944)

### particula.builder_mixin.BuilderDistributionTypeMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize distribution type mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L942-L944)

### particula.builder_mixin.BuilderDistributionTypeMixin.set_distribution_type

*method*

```python
def set_distribution_type(distribution_type: str) -> Self
```

Set the distribution type.

**Parameters**

- `distribution_type` (`str`) — One of "discrete", "continuous_pdf", or "particle_resolved".

**Returns**

- `self` (`BuilderDistributionTypeMixin`) — Self for method chaining.

**Raises**

- `ValueError` — If ``distribution_type`` is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L946-L966)

## particula.builder_mixin.BuilderLognormalMixin

*class*

```python
class BuilderLognormalMixin
```

Mixin that stores lognormal distribution properties for radius.

**Attributes**

- `mode` (`Optional[NDArray[np.float64]]`) — Mode radii in meters.
- `number_concentration` (`Optional[NDArray[np.float64]]`) — Number concentration in 1/m^3.
- `geometric_standard_deviation` (`Optional[NDArray[np.float64]]`) — Geometric standard deviation values.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L489-L568)

### particula.builder_mixin.BuilderLognormalMixin.geometric_standard_deviation

*attribute* · *instance attribute*

```python
geometric_standard_deviation: Optional[NDArray[np.float64]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L502-L502)

### particula.builder_mixin.BuilderLognormalMixin.mode

*attribute* · *instance attribute*

```python
mode: Optional[NDArray[np.float64]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L500-L500)

### particula.builder_mixin.BuilderLognormalMixin.number_concentration

*attribute* · *instance attribute*

```python
number_concentration: Optional[NDArray[np.float64]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L501-L501)

### particula.builder_mixin.BuilderLognormalMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize lognormal distribution mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L498-L502)

### particula.builder_mixin.BuilderLognormalMixin.set_geometric_standard_deviation

*method*

```python
def set_geometric_standard_deviation(geometric_standard_deviation: NDArray[np.float64], geometric_standard_deviation_units: Optional[str] = None)
```

Set the geometric standard deviation values.

**Parameters**

- `geometric_standard_deviation` (`NDArray[np.float64]`) — Dimensionless geometric std. dev.
- `geometric_standard_deviation_units` (`Optional[str]`) (default: `None`) — Optional units that are logged and ignored for interface consistency.

**Returns**

- `self` (`BuilderLognormalMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L525-L544)

### particula.builder_mixin.BuilderLognormalMixin.set_mode

*method*

```python
def set_mode(mode: NDArray[np.float64], mode_units: str)
```

Set lognormal mode radii in meters.

**Parameters**

- `mode` (`NDArray[np.float64]`) — Array of modal radius values.
- `mode_units` (`str`) — Units for ``mode``. Defaults to ``"m"``.

**Returns**

- `self` (`BuilderLognormalMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L504-L523)

### particula.builder_mixin.BuilderLognormalMixin.set_number_concentration

*method*

```python
def set_number_concentration(number_concentration: NDArray[np.float64], number_concentration_units: str)
```

Set the number concentration in 1/m^3.

**Parameters**

- `number_concentration` (`NDArray[np.float64]`) — Array of number concentration values.
- `number_concentration_units` (`str`) — Units for ``number_concentration``. Must be ``"1/m^3"`` or ``"m^-3"`` for direct storage.

**Returns**

- `self` (`BuilderLognormalMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L546-L568)

## particula.builder_mixin.BuilderMassMixin

*class*

```python
class BuilderMassMixin
```

Mixin that stores particle mass in kilograms.

**Attributes**

- `mass` (`Optional[Union[float, NDArray[np.float64]]]`) — Particle mass in kg after conversion.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L279-L309)

### particula.builder_mixin.BuilderMassMixin.mass

*attribute* · *instance attribute*

```python
mass: Optional[Union[float, NDArray[np.float64]]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L288-L288)

### particula.builder_mixin.BuilderMassMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize mass mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L286-L288)

### particula.builder_mixin.BuilderMassMixin.set_mass

*method*

```python
def set_mass(mass: Union[float, NDArray[np.float64]], mass_units: str)
```

Set the particle mass in kilograms.

**Parameters**

- `mass` (`Union[float, NDArray[np.float64]]`) — Non-negative mass value to store.
- `mass_units` (`str`) — Units for ``mass``. Defaults to ``"kg"``.

**Returns**

- `self` (`BuilderMassMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L290-L309)

## particula.builder_mixin.BuilderMolarMassMixin

*class*

```python
class BuilderMolarMassMixin
```

Mixin that stores molar mass values in kg/mol.

The mixin provides `set_molar_mass` so builders can accept molar mass inputs
in arbitrary units and normalize them to kilograms per mole.

**Attributes**

- `molar_mass` (`Optional[Union[float, NDArray[np.float64]]]`) — Stored molar mass in kg/mol.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L137-L173)

### particula.builder_mixin.BuilderMolarMassMixin.molar_mass

*attribute* · *instance attribute*

```python
molar_mass: Optional[Union[float, NDArray[np.float64]]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L149-L149)

### particula.builder_mixin.BuilderMolarMassMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize molar mass mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L147-L149)

### particula.builder_mixin.BuilderMolarMassMixin.set_molar_mass

*method*

```python
def set_molar_mass(molar_mass: Union[float, NDArray[np.float64]], molar_mass_units: str)
```

Set the molar mass in kilograms per mole.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Positive molar mass value to store.
- `molar_mass_units` (`str`) — Units for ``molar_mass``. Defaults to ``"kg/mol"``.

**Returns**

- `self` (`BuilderMolarMassMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L151-L173)

## particula.builder_mixin.BuilderParticleResolvedCountMixin

*class*

```python
class BuilderParticleResolvedCountMixin
```

Mixin that stores a particle-resolved count.

**Attributes**

- `particle_resolved_count` (`Optional[int]`) — Number of particles to resolve.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L571-L601)

### particula.builder_mixin.BuilderParticleResolvedCountMixin.particle_resolved_count

*attribute* · *instance attribute*

```python
particle_resolved_count: Optional[int] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L580-L580)

### particula.builder_mixin.BuilderParticleResolvedCountMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize particle-resolved count mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L578-L580)

### particula.builder_mixin.BuilderParticleResolvedCountMixin.set_particle_resolved_count

*method*

```python
def set_particle_resolved_count(particle_resolved_count: int, particle_resolved_count_units: Optional[str] = None)
```

Assign the particle-resolved count.

**Parameters**

- `particle_resolved_count` (`int`) — Positive integer number of particles.
- `particle_resolved_count_units` (`Optional[str]`) (default: `None`) — Optional units that are logged and ignored.

**Returns**

- `self` (`BuilderParticleResolvedCountMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L582-L601)

## particula.builder_mixin.BuilderPhaseIndexMixin

*class*

```python
class BuilderPhaseIndexMixin
```

Mixin that stores phase index assignments for species.

**Attributes**

- `phase_index` (`Optional[NDArray[np.int_]]`) — Array mapping each species to a phase identifier.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L248-L276)

### particula.builder_mixin.BuilderPhaseIndexMixin.phase_index

*attribute* · *instance attribute*

```python
phase_index: Optional[NDArray[np.int_]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L257-L257)

### particula.builder_mixin.BuilderPhaseIndexMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize phase index mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L255-L257)

### particula.builder_mixin.BuilderPhaseIndexMixin.set_phase_index

*method*

```python
def set_phase_index(phase_index: Union[Sequence[int], NDArray[np.int_]], phase_index_units: Optional[str] = None)
```

Assign phase membership indices to species.

**Parameters**

- `phase_index` (`Union[Sequence[int], NDArray[np.int_]]`) — Sequence or array of integers describing phase IDs.
- `phase_index_units` (`Optional[str]`) (default: `None`) — Optional units that are logged and ignored.

**Returns**

- `self` (`BuilderPhaseIndexMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L259-L276)

## particula.builder_mixin.BuilderPressureMixin

*class*

```python
class BuilderPressureMixin
```

Mixin that stores total gas pressure in pascals.

**Attributes**

- `pressure` (`Optional[Union[float, NDArray[np.float64]]]`) — Total pressure in Pa after conversion.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L456-L486)

### particula.builder_mixin.BuilderPressureMixin.pressure

*attribute* · *instance attribute*

```python
pressure: Optional[Union[float, NDArray[np.float64]]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L465-L465)

### particula.builder_mixin.BuilderPressureMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize pressure mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L463-L465)

### particula.builder_mixin.BuilderPressureMixin.set_pressure

*method*

```python
def set_pressure(pressure: Union[float, NDArray[np.float64]], pressure_units: str)
```

Set the total pressure in pascals.

**Parameters**

- `pressure` (`Union[float, NDArray[np.float64]]`) — Non-negative pressure value to store.
- `pressure_units` (`str`) — Units for ``pressure``. Defaults to ``"Pa"``.

**Returns**

- `self` (`BuilderPressureMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L467-L486)

## particula.builder_mixin.BuilderRadiusMixin

*class*

```python
class BuilderRadiusMixin
```

Mixin that stores particle radius in meters.

**Attributes**

- `radius` (`Optional[Union[float, NDArray[np.float64]]]`) — Particle radius in meters after conversion.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L345-L375)

### particula.builder_mixin.BuilderRadiusMixin.radius

*attribute* · *instance attribute*

```python
radius: Optional[Union[float, NDArray[np.float64]]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L354-L354)

### particula.builder_mixin.BuilderRadiusMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize radius mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L352-L354)

### particula.builder_mixin.BuilderRadiusMixin.set_radius

*method*

```python
def set_radius(radius: Union[float, NDArray[np.float64]], radius_units: str)
```

Set the particle radius in meters.

**Parameters**

- `radius` (`Union[float, NDArray[np.float64]]`) — Non-negative radius value to store.
- `radius_units` (`str`) — Units for ``radius``. Defaults to ``"m"``.

**Returns**

- `self` (`BuilderRadiusMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L356-L375)

## particula.builder_mixin.BuilderSurfaceTensionMixin

*class*

```python
class BuilderSurfaceTensionMixin
```

Mixin that stores surface tension values in N/m.

This mixin provides `set_surface_tension` so builders can accept surface
tension inputs in arbitrary units and persist them in newtons per meter.

**Attributes**

- `surface_tension` (`Optional[Union[float, NDArray[np.float64]]]`) — Surface tension in N/m after conversion.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L59-L95)

### particula.builder_mixin.BuilderSurfaceTensionMixin.surface_tension

*attribute* · *instance attribute*

```python
surface_tension: Optional[Union[float, NDArray[np.float64]]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L71-L71)

### particula.builder_mixin.BuilderSurfaceTensionMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize surface tension mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L69-L71)

### particula.builder_mixin.BuilderSurfaceTensionMixin.set_surface_tension

*method*

```python
def set_surface_tension(surface_tension: Union[float, NDArray[np.float64]], surface_tension_units: str)
```

Set the particle surface tension in newtons per meter.

**Parameters**

- `surface_tension` (`Union[float, NDArray[np.float64]]`) — Positive surface tension value to store.
- `surface_tension_units` (`str`) — Units for ``surface_tension``. Defaults to ``"N/m"`` and accepts any supported conversion.

**Returns**

- `self` (`BuilderSurfaceTensionMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L73-L95)

## particula.builder_mixin.BuilderSurfaceTensionTableMixin

*class*

```python
class BuilderSurfaceTensionTableMixin
```

Mixin that stores surface tension lookup tables in N/m.

The mixin offers `set_surface_tension_table` so builders can normalize
arrays of surface tension values to newtons per meter and retain them.

**Attributes**

- `surface_tension_table` (`Optional[NDArray[np.float64]]`) — Array of surface tension values in N/m.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L98-L134)

### particula.builder_mixin.BuilderSurfaceTensionTableMixin.surface_tension_table

*attribute* · *instance attribute*

```python
surface_tension_table: Optional[NDArray[np.float64]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L110-L110)

### particula.builder_mixin.BuilderSurfaceTensionTableMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize surface tension table mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L108-L110)

### particula.builder_mixin.BuilderSurfaceTensionTableMixin.set_surface_tension_table

*method*

```python
def set_surface_tension_table(surface_tension_table: NDArray[np.float64], surface_tension_table_units: str = 'N/m')
```

Normalize a surface tension table to newtons per meter.

**Parameters**

- `surface_tension_table` (`NDArray[np.float64]`) — Array of surface tension values.
- `surface_tension_table_units` (`str`) (default: `'N/m'`) — Units for the values. Defaults to ``"N/m"``.

**Returns**

- `self` (`BuilderSurfaceTensionTableMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L112-L134)

## particula.builder_mixin.BuilderTemperatureMixin

*class*

```python
class BuilderTemperatureMixin
```

Mixin that stores temperatures in Kelvin.

**Attributes**

- `temperature` (`Optional[float]`) — Temperature in Kelvin after conversion.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L378-L414)

### particula.builder_mixin.BuilderTemperatureMixin.temperature

*attribute* · *instance attribute*

```python
temperature: Optional[float] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L387-L387)

### particula.builder_mixin.BuilderTemperatureMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize temperature mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L385-L387)

### particula.builder_mixin.BuilderTemperatureMixin.set_temperature

*method*

```python
def set_temperature(temperature: float, temperature_units: str = 'K') -> Self
```

Set the temperature in Kelvin.

**Parameters**

- `temperature` (`float`) — Finite temperature value.
- `temperature_units` (`str`) (default: `'K'`) — Units for ``temperature``. Defaults to ``"K"``. Accepts ``"degC"``, ``"degF"``, ``"degR"``, or ``"K"``.

**Returns**

- `self` (`BuilderTemperatureMixin`) — Self for method chaining.

**Raises**

- `ValueError` — If the converted temperature is below absolute zero.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L389-L414)

## particula.builder_mixin.BuilderTemperatureTableMixin

*class*

```python
class BuilderTemperatureTableMixin
```

Mixin that stores temperature lookup tables in Kelvin.

**Attributes**

- `temperature_table` (`Optional[NDArray[np.float64]]`) — Array of temperatures in Kelvin after conversion.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L417-L453)

### particula.builder_mixin.BuilderTemperatureTableMixin.temperature_table

*attribute* · *instance attribute*

```python
temperature_table: Optional[NDArray[np.float64]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L426-L426)

### particula.builder_mixin.BuilderTemperatureTableMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize temperature table mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L424-L426)

### particula.builder_mixin.BuilderTemperatureTableMixin.set_temperature_table

*method*

```python
def set_temperature_table(temperature_table: NDArray[np.float64], temperature_table_units: str = 'K')
```

Assign a table of temperatures in Kelvin.

**Parameters**

- `temperature_table` (`NDArray[np.float64]`) — Array of temperature values.
- `temperature_table_units` (`str`) (default: `'K'`) — Units for the values. Defaults to ``"K"``.

**Returns**

- `self` (`BuilderTemperatureTableMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L428-L453)

## particula.builder_mixin.BuilderVolumeMixin

*class*

```python
class BuilderVolumeMixin
```

Mixin that stores particle volume in cubic meters.

**Attributes**

- `volume` (`Optional[Union[float, NDArray[np.float64]]]`) — Particle volume in m^3 after conversion.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L312-L342)

### particula.builder_mixin.BuilderVolumeMixin.volume

*attribute* · *instance attribute*

```python
volume: Optional[Union[float, NDArray[np.float64]]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L321-L321)

### particula.builder_mixin.BuilderVolumeMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize volume mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L319-L321)

### particula.builder_mixin.BuilderVolumeMixin.set_volume

*method*

```python
def set_volume(volume: Union[float, NDArray[np.float64]], volume_units: str)
```

Set the particle volume in cubic meters.

**Parameters**

- `volume` (`Union[float, NDArray[np.float64]]`) — Non-negative volume value to store.
- `volume_units` (`str`) — Units for ``volume``. Defaults to ``"m^3"``.

**Returns**

- `self` (`BuilderVolumeMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L323-L342)

## particula.builder_mixin.BuilderWallEddyDiffusivityMixin

*class*

```python
class BuilderWallEddyDiffusivityMixin
```

Mixin for setting wall eddy diffusivity.

Adds ``set_wall_eddy_diffusivity`` for builders that require wall eddy
diffusivity and stores the value in 1/s.

**Attributes**

- `wall_eddy_diffusivity` (`Optional[float]`) — Wall eddy diffusivity in 1/s.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L604-L648)

### particula.builder_mixin.BuilderWallEddyDiffusivityMixin.wall_eddy_diffusivity

*attribute* · *instance attribute*

```python
wall_eddy_diffusivity: Optional[float] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L616-L616)

### particula.builder_mixin.BuilderWallEddyDiffusivityMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize wall eddy diffusivity mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L614-L616)

### particula.builder_mixin.BuilderWallEddyDiffusivityMixin.set_wall_eddy_diffusivity

*method*

```python
def set_wall_eddy_diffusivity(wall_eddy_diffusivity: float, wall_eddy_diffusivity_units: str = '1/s') -> BuilderWallEddyDiffusivityMixin
```

Set wall eddy diffusivity in 1/s.

**Parameters**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity value.
- `wall_eddy_diffusivity_units` (`str`) (default: `'1/s'`) — Units for the value. Defaults to "1/s".

**Returns**

- `self` (`BuilderWallEddyDiffusivityMixin`) — Self for method chaining.

**Raises**

- `ValueError` — If ``wall_eddy_diffusivity`` is not positive.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L618-L648)

## particula.builder_mixin.BuilderWallElectricFieldMixin

*class*

```python
class BuilderWallElectricFieldMixin
```

Mixin that stores wall electric field magnitudes in V/m.

Supports optional drift terms for charged wall loss strategies.
Accepts a scalar magnitude for spherical chambers or a three-component
tuple for rectangular geometries. A value of 0.0 disables field-driven
drift.

**Attributes**

- `wall_electric_field` (`Union[float, Tuple[float, float, float]]`) — Electric field magnitude (scalar or tuple) in V/m.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L823-L929)

### particula.builder_mixin.BuilderWallElectricFieldMixin.wall_electric_field

*attribute* · *instance attribute*

```python
wall_electric_field: Union[float, Tuple[float, float, float]] = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L837-L837)

### particula.builder_mixin.BuilderWallElectricFieldMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize wall electric field mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L835-L837)

### particula.builder_mixin.BuilderWallElectricFieldMixin._validate_wall_electric_field

*method*

```python
def _validate_wall_electric_field(wall_electric_field: Union[float, Tuple[float, float, float]]) -> Union[float, Tuple[float, float, float]]
```

Validate and normalize wall electric field inputs.

**Parameters**

- `wall_electric_field` (`Union[float, Tuple[float, float, float]]`) — Scalar magnitude or ``(Ex, Ey, Ez)`` tuple.

**Returns**

- (`Union[float, Tuple[float, float, float]]`) — Union[float, Tuple[float, float, float]]: Validated scalar or tuple.

**Raises**

- `ValueError` — If tuple length is not three or entries are not finite.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L839-L864)

### particula.builder_mixin.BuilderWallElectricFieldMixin.set_wall_electric_field

*method*

```python
def set_wall_electric_field(wall_electric_field: Union[float, Tuple[float, float, float]], wall_electric_field_units: str = 'V/m') -> BuilderWallElectricFieldMixin
```

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

## particula.builder_mixin.BuilderWallPotentialMixin

*class*

```python
class BuilderWallPotentialMixin
```

Mixin that stores electrostatic wall potential in volts.

Stores electrostatic wall potential used by charged wall loss strategies.
A zero potential still permits image-charge enhancement when particle
charge is non-zero.

**Attributes**

- `wall_potential` (`float`) — Electrostatic wall potential in volts.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L782-L820)

### particula.builder_mixin.BuilderWallPotentialMixin.wall_potential

*attribute* · *instance attribute*

```python
wall_potential: float = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L795-L795)

### particula.builder_mixin.BuilderWallPotentialMixin.__init__

*method*

```python
def __init__() -> None
```

Initialize wall potential mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L793-L795)

### particula.builder_mixin.BuilderWallPotentialMixin.set_wall_potential

*method*

```python
def set_wall_potential(wall_potential: float, wall_potential_units: str = 'V') -> BuilderWallPotentialMixin
```

Set wall potential in volts.

**Parameters**

- `wall_potential` (`float`) — Electrostatic wall potential in volts. Zero keeps the chamber neutral but still allows image-charge effects for charged particles.
- `wall_potential_units` (`str`) (default: `'V'`) — Units for ``wall_potential``. Defaults to "V".

**Returns**

- `self` (`BuilderWallPotentialMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L797-L820)
