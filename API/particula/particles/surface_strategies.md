# particula.particles.surface_strategies

*module*

Strategies for surface effects on particles.

Provides classes for calculating effective surface tension and the
Kelvin effect for species in particulate phases. Future expansions
may include an organic film strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L1-L1)

## particula.particles.surface_strategies.SurfaceStrategy

*class*

```python
class SurfaceStrategy(ABC)
```

Bases: `ABC`

Abstract base class for surface strategies.

Implements methods for calculating effective surface tension, density,
and the Kelvin effect in particulate phases.

Methods:
- effective_surface_tension : Calculate the effective surface tension.
- effective_density : Calculate the effective density.
- get_name : Return the type of the surface strategy.
- kelvin_radius : Calculate the Kelvin radius for curvature effects.
- kelvin_term : Calculate the exponential Kelvin term for vapor pressure.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L166-L302)

### particula.particles.surface_strategies.SurfaceStrategy.surface_tension

*attribute* · *instance attribute*

```python
surface_tension = np.asarray(surface_tension, dtype=np.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L187-L187)

### particula.particles.surface_strategies.SurfaceStrategy.surface_tension_table

*attribute* · *instance attribute*

```python
surface_tension_table = np.asarray(surface_tension_table, dtype=np.float64) if surface_tension_table is not None else None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L188-L192)

### particula.particles.surface_strategies.SurfaceStrategy.temperature_table

*attribute* · *instance attribute*

```python
temperature_table = np.asarray(temperature_table, dtype=np.float64) if temperature_table is not None else None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L193-L197)

### particula.particles.surface_strategies.SurfaceStrategy.__init__

*method*

```python
def __init__(surface_tension: Union[float, NDArray[np.float64]] = 0.072, surface_tension_table: Optional[NDArray[np.float64]] = None, temperature_table: Optional[NDArray[np.float64]] = None)
```

Initialize the surface strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L180-L197)

### particula.particles.surface_strategies.SurfaceStrategy.effective_surface_tension

*method* · *abstract*

```python
def effective_surface_tension(mass_concentration: Union[float, NDArray[np.float64]], temperature: Optional[float] = None) -> Union[float, NDArray[np.float64]]
```

Calculate the effective surface tension of the species mixture.

**Parameters**

- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.
- `temperature` (`Optional[float]`) (default: `None`) — Optional temperature dependence in K.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Effective surface tension in N/m.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L199-L213)

### particula.particles.surface_strategies.SurfaceStrategy.get_density

*method* · *abstract*

```python
def get_density() -> Union[float, NDArray[np.float64]]
```

Get density of the species mixture.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — density in kg/m^3.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L215-L221)

### particula.particles.surface_strategies.SurfaceStrategy.get_name

*method*

```python
def get_name() -> str
```

Return the type of the surface strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L223-L225)

### particula.particles.surface_strategies.SurfaceStrategy.kelvin_radius

*method*

```python
def kelvin_radius(molar_mass: Union[float, NDArray[np.float64]], mass_concentration: Union[float, NDArray[np.float64]], temperature: float) -> Union[float, NDArray[np.float64]]
```

Calculate the Kelvin radius, which sets the curvature effect on vapor
pressure.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the species in kg/mol.
- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.
- `temperature` (`float`) — Temperature of the system in K.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Kelvin radius in meters.

> **References**
>
> - r = 2 × surface_tension × molar_mass / (R × T × density)
>   [Kelvin Equation](https://en.wikipedia.org/wiki/Kelvin_equation)

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L227-L253)

### particula.particles.surface_strategies.SurfaceStrategy.kelvin_term

*method*

```python
def kelvin_term(radius: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], mass_concentration: Union[float, NDArray[np.float64]], temperature: float) -> Union[float, NDArray[np.float64]]
```

Calculate the exponential Kelvin term that adjusts vapor pressure.

**Parameters**

- `radius` (`Union[float, NDArray[np.float64]]`) — Particle radius in meters.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the species in kg/mol.
- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.
- `temperature` (`float`) — Temperature of the system in K.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Factor by which vapor pressure is increased.

> **References**
>
> - P_eff = P_sat × exp(kelvin_radius / particle_radius)
>   [Kelvin Equation](https://en.wikipedia.org/wiki/Kelvin_equation)

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L255-L280)

### particula.particles.surface_strategies.SurfaceStrategy.update_surface_tension

*method*

```python
def update_surface_tension(temperature: float) -> None
```

Update the surface tension attribute based on the given temperature.

If a temperature-dependent surface tension table is provided,
it interpolates the surface tension value for the specified
temperature.

**Parameters**

- `temperature` (`float`) — Temperature in K to update the surface tension.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L282-L302)

## particula.particles.surface_strategies.SurfaceStrategyMass

*class*

```python
class SurfaceStrategyMass(SurfaceStrategy)
```

Bases: `particula.particles.surface_strategies.SurfaceStrategy`

Surface tension and density based on mass-fraction weighting.

**Attributes**

- `surface_tension` — Surface tension array or scalar in N/m.
- `density` — Density array or scalar in kg/m^3.
- `phase_index` — Optional array indicating phase indices for species. Example: [0, 1, 1] → three species in two phases (first in phase 0, last two in phase 1).

References:
- [Mass Fraction](https://en.wikipedia.org/wiki/Mass_fraction_(chemistry))

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L386-L453)

### particula.particles.surface_strategies.SurfaceStrategyMass.density

*attribute* · *instance attribute*

```python
density = density
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L419-L419)

### particula.particles.surface_strategies.SurfaceStrategyMass.phase_index

*attribute* · *instance attribute*

```python
phase_index = None if phase_index is None else np.array(phase_index, dtype=int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L420-L422)

### particula.particles.surface_strategies.SurfaceStrategyMass.surface_tension

*attribute* · *instance attribute*

```python
surface_tension = np.asarray(surface_tension, dtype=np.float64)
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L187-L187)

### particula.particles.surface_strategies.SurfaceStrategyMass.surface_tension_table

*attribute* · *instance attribute*

```python
surface_tension_table = np.asarray(surface_tension_table, dtype=np.float64) if surface_tension_table is not None else None
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L188-L192)

### particula.particles.surface_strategies.SurfaceStrategyMass.temperature_table

*attribute* · *instance attribute*

```python
temperature_table = np.asarray(temperature_table, dtype=np.float64) if temperature_table is not None else None
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L193-L197)

### particula.particles.surface_strategies.SurfaceStrategyMass.__init__

*method*

```python
def __init__(surface_tension: Union[float, NDArray[np.float64]] = 0.072, density: Union[float, NDArray[np.float64]] = 1000, phase_index: Optional[Union[Sequence[int], NDArray[np.int_]]] = None, surface_tension_table: Optional[NDArray[np.float64]] = None, temperature_table: Optional[NDArray[np.float64]] = None)
```

Initialize the SurfaceStrategyMass.

Sets up the mass-fraction-based surface strategy with surface
tension and density parameters for calculating effective surface
properties.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L400-L422)

### particula.particles.surface_strategies.SurfaceStrategyMass.effective_surface_tension

*method*

```python
def effective_surface_tension(mass_concentration: Union[float, NDArray[np.float64]], temperature: Optional[float] = None) -> Union[float, NDArray[np.float64]]
```

Calculate effective surface tension using mass-fraction weighting.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Effective surface tension in N/m.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L424-L445)

### particula.particles.surface_strategies.SurfaceStrategyMass.get_density

*method*

```python
def get_density() -> Union[float, NDArray[np.float64]]
```

Get density of the species mixture.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Density in kg/m^3.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L447-L453)

### particula.particles.surface_strategies.SurfaceStrategyMass.get_name

*method*

```python
def get_name() -> str
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

Return the type of the surface strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L223-L225)

### particula.particles.surface_strategies.SurfaceStrategyMass.kelvin_radius

*method*

```python
def kelvin_radius(molar_mass: Union[float, NDArray[np.float64]], mass_concentration: Union[float, NDArray[np.float64]], temperature: float) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

Calculate the Kelvin radius, which sets the curvature effect on vapor
pressure.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the species in kg/mol.
- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.
- `temperature` (`float`) — Temperature of the system in K.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Kelvin radius in meters.

> **References**
>
> - r = 2 × surface_tension × molar_mass / (R × T × density)
>   [Kelvin Equation](https://en.wikipedia.org/wiki/Kelvin_equation)

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L227-L253)

### particula.particles.surface_strategies.SurfaceStrategyMass.kelvin_term

*method*

```python
def kelvin_term(radius: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], mass_concentration: Union[float, NDArray[np.float64]], temperature: float) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

Calculate the exponential Kelvin term that adjusts vapor pressure.

**Parameters**

- `radius` (`Union[float, NDArray[np.float64]]`) — Particle radius in meters.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the species in kg/mol.
- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.
- `temperature` (`float`) — Temperature of the system in K.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Factor by which vapor pressure is increased.

> **References**
>
> - P_eff = P_sat × exp(kelvin_radius / particle_radius)
>   [Kelvin Equation](https://en.wikipedia.org/wiki/Kelvin_equation)

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L255-L280)

### particula.particles.surface_strategies.SurfaceStrategyMass.update_surface_tension

*method*

```python
def update_surface_tension(temperature: float) -> None
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

Update the surface tension attribute based on the given temperature.

If a temperature-dependent surface tension table is provided,
it interpolates the surface tension value for the specified
temperature.

**Parameters**

- `temperature` (`float`) — Temperature in K to update the surface tension.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L282-L302)

## particula.particles.surface_strategies.SurfaceStrategyMolar

*class*

```python
class SurfaceStrategyMolar(SurfaceStrategy)
```

Bases: `particula.particles.surface_strategies.SurfaceStrategy`

Surface tension and density based on mole-fraction weighting.

**Attributes**

- `surface_tension` — Surface tension array or scalar in N/m.
- `density` — Density array or scalar in kg/m^3.
- `molar_mass` — Molar mass array or scalar in kg/mol.
- `phase_index` — Optional array indicating phase indices for species. For example, [0, 1, 1] for two phases, where the first species belongs to phase 0 and the next two to phase 1.
- `surface_tension_table` — Optional 2D array for temperature-dependent surface tension values.
- `temperature_table` — Optional 1D array for temperature values corresponding to the surface tension table.

> **References**
>
> - [Mole Fraction](https://en.wikipedia.org/wiki/Mole_fraction)

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L306-L383)

### particula.particles.surface_strategies.SurfaceStrategyMolar.density

*attribute* · *instance attribute*

```python
density = density
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L345-L345)

### particula.particles.surface_strategies.SurfaceStrategyMolar.molar_mass

*attribute* · *instance attribute*

```python
molar_mass = molar_mass
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L346-L346)

### particula.particles.surface_strategies.SurfaceStrategyMolar.phase_index

*attribute* · *instance attribute*

```python
phase_index = None if phase_index is None else np.array(phase_index, dtype=int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L347-L349)

### particula.particles.surface_strategies.SurfaceStrategyMolar.surface_tension

*attribute* · *instance attribute*

```python
surface_tension = np.asarray(surface_tension, dtype=np.float64)
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L187-L187)

### particula.particles.surface_strategies.SurfaceStrategyMolar.surface_tension_table

*attribute* · *instance attribute*

```python
surface_tension_table = np.asarray(surface_tension_table, dtype=np.float64) if surface_tension_table is not None else None
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L188-L192)

### particula.particles.surface_strategies.SurfaceStrategyMolar.temperature_table

*attribute* · *instance attribute*

```python
temperature_table = np.asarray(temperature_table, dtype=np.float64) if temperature_table is not None else None
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L193-L197)

### particula.particles.surface_strategies.SurfaceStrategyMolar.__init__

*method*

```python
def __init__(surface_tension: Union[float, NDArray[np.float64]] = 0.072, density: Union[float, NDArray[np.float64]] = 1000, molar_mass: Union[float, NDArray[np.float64]] = 0.01815, phase_index: Optional[Union[Sequence[int], NDArray[np.int_]]] = None, surface_tension_table: Optional[NDArray[np.float64]] = None, temperature_table: Optional[NDArray[np.float64]] = None)
```

Initialize the SurfaceStrategyMolar.

Sets up the molar-fraction-based surface strategy with surface
tension, density, and molar mass parameters for calculating
effective surface properties.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L325-L349)

### particula.particles.surface_strategies.SurfaceStrategyMolar.effective_surface_tension

*method*

```python
def effective_surface_tension(mass_concentration: Union[float, NDArray[np.float64]], temperature: Optional[float] = None) -> Union[float, NDArray[np.float64]]
```

Calculate effective surface tension using mole-fraction weighting.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Effective surface tension in N/m.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L351-L375)

### particula.particles.surface_strategies.SurfaceStrategyMolar.get_density

*method*

```python
def get_density() -> Union[float, NDArray[np.float64]]
```

Get density of the species mixture.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Density in kg/m^3.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L377-L383)

### particula.particles.surface_strategies.SurfaceStrategyMolar.get_name

*method*

```python
def get_name() -> str
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

Return the type of the surface strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L223-L225)

### particula.particles.surface_strategies.SurfaceStrategyMolar.kelvin_radius

*method*

```python
def kelvin_radius(molar_mass: Union[float, NDArray[np.float64]], mass_concentration: Union[float, NDArray[np.float64]], temperature: float) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

Calculate the Kelvin radius, which sets the curvature effect on vapor
pressure.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the species in kg/mol.
- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.
- `temperature` (`float`) — Temperature of the system in K.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Kelvin radius in meters.

> **References**
>
> - r = 2 × surface_tension × molar_mass / (R × T × density)
>   [Kelvin Equation](https://en.wikipedia.org/wiki/Kelvin_equation)

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L227-L253)

### particula.particles.surface_strategies.SurfaceStrategyMolar.kelvin_term

*method*

```python
def kelvin_term(radius: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], mass_concentration: Union[float, NDArray[np.float64]], temperature: float) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

Calculate the exponential Kelvin term that adjusts vapor pressure.

**Parameters**

- `radius` (`Union[float, NDArray[np.float64]]`) — Particle radius in meters.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the species in kg/mol.
- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.
- `temperature` (`float`) — Temperature of the system in K.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Factor by which vapor pressure is increased.

> **References**
>
> - P_eff = P_sat × exp(kelvin_radius / particle_radius)
>   [Kelvin Equation](https://en.wikipedia.org/wiki/Kelvin_equation)

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L255-L280)

### particula.particles.surface_strategies.SurfaceStrategyMolar.update_surface_tension

*method*

```python
def update_surface_tension(temperature: float) -> None
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

Update the surface tension attribute based on the given temperature.

If a temperature-dependent surface tension table is provided,
it interpolates the surface tension value for the specified
temperature.

**Parameters**

- `temperature` (`float`) — Temperature in K to update the surface tension.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L282-L302)

## particula.particles.surface_strategies.SurfaceStrategyVolume

*class*

```python
class SurfaceStrategyVolume(SurfaceStrategy)
```

Bases: `particula.particles.surface_strategies.SurfaceStrategy`

Surface tension and density based on volume-fraction weighting.

**Attributes**

- `surface_tension` — Surface tension array or scalar in N/m.
- `density` — Density array or scalar in kg/m^3.
- `phase_index` — Optional array indicating phase indices for species. Example: [0, 1, 1] → three species in two phases (first in phase 0, last two in phase 1).

> **References**
>
> - [Volume Fraction](https://en.wikipedia.org/wiki/Volume_fraction)

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L456-L526)

### particula.particles.surface_strategies.SurfaceStrategyVolume.density

*attribute* · *instance attribute*

```python
density = density
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L489-L489)

### particula.particles.surface_strategies.SurfaceStrategyVolume.phase_index

*attribute* · *instance attribute*

```python
phase_index = None if phase_index is None else np.array(phase_index, dtype=int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L490-L492)

### particula.particles.surface_strategies.SurfaceStrategyVolume.surface_tension

*attribute* · *instance attribute*

```python
surface_tension = np.asarray(surface_tension, dtype=np.float64)
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L187-L187)

### particula.particles.surface_strategies.SurfaceStrategyVolume.surface_tension_table

*attribute* · *instance attribute*

```python
surface_tension_table = np.asarray(surface_tension_table, dtype=np.float64) if surface_tension_table is not None else None
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L188-L192)

### particula.particles.surface_strategies.SurfaceStrategyVolume.temperature_table

*attribute* · *instance attribute*

```python
temperature_table = np.asarray(temperature_table, dtype=np.float64) if temperature_table is not None else None
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L193-L197)

### particula.particles.surface_strategies.SurfaceStrategyVolume.__init__

*method*

```python
def __init__(surface_tension: Union[float, NDArray[np.float64]] = 0.072, density: Union[float, NDArray[np.float64]] = 1000, phase_index: Optional[Union[Sequence[int], NDArray[np.int_]]] = None, surface_tension_table: Optional[NDArray[np.float64]] = None, temperature_table: Optional[NDArray[np.float64]] = None)
```

Initialize the SurfaceStrategyVolume.

Sets up the volume-fraction-based surface strategy with surface
tension and density parameters for calculating effective surface
properties.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L470-L492)

### particula.particles.surface_strategies.SurfaceStrategyVolume.effective_surface_tension

*method*

```python
def effective_surface_tension(mass_concentration: Union[float, NDArray[np.float64]], temperature: Optional[float] = None) -> Union[float, NDArray[np.float64]]
```

Calculate effective surface tension using volume-fraction weighting.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Effective surface tension in N/m.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L494-L518)

### particula.particles.surface_strategies.SurfaceStrategyVolume.get_density

*method*

```python
def get_density() -> Union[float, NDArray[np.float64]]
```

Get density of the species mixture.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Density in kg/m^3.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L520-L526)

### particula.particles.surface_strategies.SurfaceStrategyVolume.get_name

*method*

```python
def get_name() -> str
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

Return the type of the surface strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L223-L225)

### particula.particles.surface_strategies.SurfaceStrategyVolume.kelvin_radius

*method*

```python
def kelvin_radius(molar_mass: Union[float, NDArray[np.float64]], mass_concentration: Union[float, NDArray[np.float64]], temperature: float) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

Calculate the Kelvin radius, which sets the curvature effect on vapor
pressure.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the species in kg/mol.
- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.
- `temperature` (`float`) — Temperature of the system in K.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Kelvin radius in meters.

> **References**
>
> - r = 2 × surface_tension × molar_mass / (R × T × density)
>   [Kelvin Equation](https://en.wikipedia.org/wiki/Kelvin_equation)

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L227-L253)

### particula.particles.surface_strategies.SurfaceStrategyVolume.kelvin_term

*method*

```python
def kelvin_term(radius: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], mass_concentration: Union[float, NDArray[np.float64]], temperature: float) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

Calculate the exponential Kelvin term that adjusts vapor pressure.

**Parameters**

- `radius` (`Union[float, NDArray[np.float64]]`) — Particle radius in meters.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the species in kg/mol.
- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.
- `temperature` (`float`) — Temperature of the system in K.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Factor by which vapor pressure is increased.

> **References**
>
> - P_eff = P_sat × exp(kelvin_radius / particle_radius)
>   [Kelvin Equation](https://en.wikipedia.org/wiki/Kelvin_equation)

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L255-L280)

### particula.particles.surface_strategies.SurfaceStrategyVolume.update_surface_tension

*method*

```python
def update_surface_tension(temperature: float) -> None
```

Inherited from: `particula.particles.surface_strategies.SurfaceStrategy`

Update the surface tension attribute based on the given temperature.

If a temperature-dependent surface tension table is provided,
it interpolates the surface tension value for the specified
temperature.

**Parameters**

- `temperature` (`float`) — Temperature in K to update the surface tension.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L282-L302)

## particula.particles.surface_strategies._as_2d

*function*

```python
def _as_2d(array: NDArray[np.float64]) -> tuple[NDArray[np.float64], bool]
```

Promote *array* to 2-D (n_bins, n_species).

Returns:
-------
arr2d : the reshaped array
was_1d : True if the original input was 1-D (hence single-bin)

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L27-L39)

## particula.particles.surface_strategies._broadcast_weights

*function*

```python
def _broadcast_weights(weights: NDArray[np.float64], target_shape: tuple[int, int]) -> NDArray[np.float64]
```

Return *weights* with shape exactly equal to *target_shape*
(n_bins, n_species).  Accepts 1-D (n_species,) or 2-D inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L71-L102)

## particula.particles.surface_strategies._interp_surface_tension

*function*

```python
def _interp_surface_tension(temperature: float, surface_tension_table: NDArray[np.float64], temperature_table: NDArray[np.float64]) -> Union[float, NDArray[np.float64]]
```

Interpolate *surface_tension_table* at *temperature*.
Handles 1-D (single species) and 2-D (one column per species) tables.

Returns a scalar (for 1-D) or 1-D array (for 2-D).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L42-L68)

## particula.particles.surface_strategies._weighted_average_by_phase

*function*

```python
def _weighted_average_by_phase(values: NDArray[np.float64], weights: NDArray[np.float64], phase_index: NDArray[np.int_]) -> NDArray[np.float64]
```

Return an array where each element equals the phase-averaged *values*
using *weights* as weighting factors.

For every unique entry in `phase_index` the weighted-average is
computed and broadcast back onto all members of that phase.

**Parameters**

- `values` (`NDArray[np.float64]`) — Array of values, shape (n_bins, n_species) or (n_species,)
- `weights` (`NDArray[np.float64]`) — Array of weights corresponding to each species, shape (n_species,) **or** (n_bins, n_species).  A 1-D array is broadcast across the n_bins axis.
- `phase_index` (`NDArray[np.int_]`) — Array indicating the phase index for each species, shape (n_species,).

**Returns**

- `averaged` (`NDArray[np.float64]`) — Array of averaged values, same shape as *values* input. If the weight sum for a bin/phase is zero, the arithmetic mean is used.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/surface_strategies.py#L105-L163)
