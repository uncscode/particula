# particula.particles.activity_strategies

*module*

Common interface for activity calculations.

Class strategies for activities and vapor pressure over mixture of liquids
surface Using Raoult's Law, and strategies ideal, non-ideal, kappa hygroscopic
parameterizations.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L1-L1)

## particula.particles.activity_strategies.ActivityBAT

*class*

```python
class ActivityBAT(ActivityNonIdealBinary)
```

Bases: `particula.particles.activity_strategies.ActivityNonIdealBinary`

Homogeneous binary BAT activities in [water, organic] material order.

Configure organic properties as for ActivityNonIdealBinary. Both species
activities enter the kinetic condensation surface-pressure calculation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L355-L383)

### particula.particles.activity_strategies.ActivityBAT.density

*attribute* · *instance attribute*

```python
density = density
```

Inherited from: `particula.particles.activity_strategies.ActivityNonIdealBinary`

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L291-L291)

### particula.particles.activity_strategies.ActivityBAT.functional_group

*attribute* · *instance attribute*

```python
functional_group = functional_group
```

Inherited from: `particula.particles.activity_strategies.ActivityNonIdealBinary`

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L292-L292)

### particula.particles.activity_strategies.ActivityBAT.molar_mass_g

*attribute* · *instance attribute*

```python
molar_mass_g = self.molar_mass_kg * 1000.0
```

Inherited from: `particula.particles.activity_strategies.ActivityNonIdealBinary`

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L289-L289)

### particula.particles.activity_strategies.ActivityBAT.molar_mass_kg

*attribute* · *instance attribute*

```python
molar_mass_kg = float(molar_mass)
```

Inherited from: `particula.particles.activity_strategies.ActivityNonIdealBinary`

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L288-L288)

### particula.particles.activity_strategies.ActivityBAT.oxygen2carbon

*attribute* · *instance attribute*

```python
oxygen2carbon = oxygen2carbon
```

Inherited from: `particula.particles.activity_strategies.ActivityNonIdealBinary`

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L290-L290)

### particula.particles.activity_strategies.ActivityBAT.provider_interface_version

*attribute* · *class attribute* · *instance attribute*

```python
provider_interface_version = 1
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L56-L56)

### particula.particles.activity_strategies.ActivityBAT.activity

*method*

```python
def activity(mass_concentration)
```

Return [water, organic] activities on the final material axis.

Masses share a common mass basis. Empty rows represent unused slots
and contribute zero surface pressure to both lanes.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L362-L379)

### particula.particles.activity_strategies.ActivityBAT.get_name

*method*

```python
def get_name() -> str
```

Return the two-species strategy identifier.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L381-L383)

### particula.particles.activity_strategies.ActivityBAT.__init__

*method*

```python
def __init__(molar_mass: float, oxygen2carbon: float, density: float, functional_group: Optional[Union[str, List[str]]] = None) -> None
```

Inherited from: `particula.particles.activity_strategies.ActivityNonIdealBinary`

Initialize the binary non-ideal activity strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L280-L292)

### particula.particles.activity_strategies.ActivityBAT._activity_pair

*method*

```python
def _activity_pair(mass_concentration)
```

Inherited from: `particula.particles.activity_strategies.ActivityNonIdealBinary`

Evaluate water and organic activities at the same composition.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L312-L348)

### particula.particles.activity_strategies.ActivityBAT.neutral_activities

*method*

```python
def neutral_activities(masses, temperature, targets, active) -> NDArray[np.float64]
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

Return dimensionless activity in requested material order.

Version 1 receives full particle-by-material masses (kg), the current
temperature in K and a boolean particle occupancy mask. Existing
strategies retain their full-composition unit-pressure evaluation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L58-L74)

### particula.particles.activity_strategies.ActivityBAT.partial_pressure

*method*

```python
def partial_pressure(pure_vapor_pressure: Union[float, NDArray[np.float64]], mass_concentration: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

Calculate the vapor pressure of species in the particle phase.

**Parameters**

- `pure_vapor_pressure` (`Union[float, NDArray[np.float64]]`) — Pure vapor pressure of the species in Pa.
- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Vapor pressure of the particle in Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L93-L110)

## particula.particles.activity_strategies.ActivityIdealMass

*class*

```python
class ActivityIdealMass(ActivityStrategy)
```

Bases: `particula.particles.activity_strategies.ActivityStrategy`

Calculate ideal activity based on mass fractions (Raoult's Law).

- activity : Computes activity from mass concentration,
    treating mass fractions as ideal.

**Examples**

```py title="Example Usage"
import particula as par
strategy = par.particles.ActivityIdealMass()
a = strategy.activity([0.5, 1.0, 1.5])
# a -> ...
```

References:
- "Raoult's Law,"
    [Wikipedia](https://en.wikipedia.org/wiki/Raoult%27s_law).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L161-L192)

### particula.particles.activity_strategies.ActivityIdealMass.provider_interface_version

*attribute* · *class attribute* · *instance attribute*

```python
provider_interface_version = 1
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L56-L56)

### particula.particles.activity_strategies.ActivityIdealMass.activity

*method*

```python
def activity(mass_concentration: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the activity of a species based on mass concentration.

**Parameters**

- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Activity of the species, unitless.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L181-L192)

### particula.particles.activity_strategies.ActivityIdealMass.get_name

*method*

```python
def get_name() -> str
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

Return the type of the activity strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L89-L91)

### particula.particles.activity_strategies.ActivityIdealMass.neutral_activities

*method*

```python
def neutral_activities(masses, temperature, targets, active) -> NDArray[np.float64]
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

Return dimensionless activity in requested material order.

Version 1 receives full particle-by-material masses (kg), the current
temperature in K and a boolean particle occupancy mask. Existing
strategies retain their full-composition unit-pressure evaluation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L58-L74)

### particula.particles.activity_strategies.ActivityIdealMass.partial_pressure

*method*

```python
def partial_pressure(pure_vapor_pressure: Union[float, NDArray[np.float64]], mass_concentration: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

Calculate the vapor pressure of species in the particle phase.

**Parameters**

- `pure_vapor_pressure` (`Union[float, NDArray[np.float64]]`) — Pure vapor pressure of the species in Pa.
- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Vapor pressure of the particle in Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L93-L110)

## particula.particles.activity_strategies.ActivityIdealMolar

*class*

```python
class ActivityIdealMolar(ActivityStrategy)
```

Bases: `particula.particles.activity_strategies.ActivityStrategy`

Calculate ideal activity based on mole fractions (Raoult's Law).

**Attributes**

- `molar_mass` — Molar mass of the species in kg/mol.

**Functions**

- `- activity ` — Computes ideal activity from mass concentration
and molar mass.

**Examples**

```py title="Example Usage"
import particula as par
strategy = par.particles.ActivityIdealMolar(molar_mass=0.018)
# mass_concentration in kg/m^3
a = strategy.activity(np.array([1.2, 2.5, 3.0]))
# a -> ...
```

> **References**
>
> - "Raoult's Law,"
> [Wikipedia](https://en.wikipedia.org/wiki/Raoult%27s_law).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L113-L158)

### particula.particles.activity_strategies.ActivityIdealMolar.molar_mass

*attribute* · *instance attribute*

```python
molar_mass = molar_mass
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L143-L143)

### particula.particles.activity_strategies.ActivityIdealMolar.provider_interface_version

*attribute* · *class attribute* · *instance attribute*

```python
provider_interface_version = 1
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L56-L56)

### particula.particles.activity_strategies.ActivityIdealMolar.__init__

*method*

```python
def __init__(molar_mass: Union[float, NDArray[np.float64]] = 0.0)
```

Initialize the ActivityIdealMolar strategy.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) (default: `0.0`) — Molar mass in kg/mol.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L137-L143)

### particula.particles.activity_strategies.ActivityIdealMolar.activity

*method*

```python
def activity(mass_concentration: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the activity of a species based on mass concentration.

**Parameters**

- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Activity of the species, unitless.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L145-L158)

### particula.particles.activity_strategies.ActivityIdealMolar.get_name

*method*

```python
def get_name() -> str
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

Return the type of the activity strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L89-L91)

### particula.particles.activity_strategies.ActivityIdealMolar.neutral_activities

*method*

```python
def neutral_activities(masses, temperature, targets, active) -> NDArray[np.float64]
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

Return dimensionless activity in requested material order.

Version 1 receives full particle-by-material masses (kg), the current
temperature in K and a boolean particle occupancy mask. Existing
strategies retain their full-composition unit-pressure evaluation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L58-L74)

### particula.particles.activity_strategies.ActivityIdealMolar.partial_pressure

*method*

```python
def partial_pressure(pure_vapor_pressure: Union[float, NDArray[np.float64]], mass_concentration: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

Calculate the vapor pressure of species in the particle phase.

**Parameters**

- `pure_vapor_pressure` (`Union[float, NDArray[np.float64]]`) — Pure vapor pressure of the species in Pa.
- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Vapor pressure of the particle in Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L93-L110)

## particula.particles.activity_strategies.ActivityIdealVolume

*class*

```python
class ActivityIdealVolume(ActivityStrategy)
```

Bases: `particula.particles.activity_strategies.ActivityStrategy`

Calculate ideal activity based on volume fractions (Raoult's Law).

**Attributes**

- `density` — The density of the species in kg/m^3, used to derive volume fractions from mass concentrations.

Methods:
- activity : Computes activity from mass concentration and density.

**Examples**

```py title="Example Usage"
strategy = ActivityIdealVolume(density=1000.0)
a = strategy.activity(2.5)
# a -> ...
```

References:
- "Raoult's Law,"
    [Wikipedia](https://en.wikipedia.org/wiki/Raoult%27s_law).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L195-L238)

### particula.particles.activity_strategies.ActivityIdealVolume.density

*attribute* · *instance attribute*

```python
density = density
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L223-L223)

### particula.particles.activity_strategies.ActivityIdealVolume.provider_interface_version

*attribute* · *class attribute* · *instance attribute*

```python
provider_interface_version = 1
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L56-L56)

### particula.particles.activity_strategies.ActivityIdealVolume.__init__

*method*

```python
def __init__(density: Union[float, NDArray[np.float64]] = 0.0)
```

Initialize the ActivityIdealVolume strategy.

**Parameters**

- `density` (`Union[float, NDArray[np.float64]]`) (default: `0.0`) — Density of the species in kg/m^3.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L217-L223)

### particula.particles.activity_strategies.ActivityIdealVolume.activity

*method*

```python
def activity(mass_concentration: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the activity of a species based on mass concentration.

**Parameters**

- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Activity of the species, unitless.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L225-L238)

### particula.particles.activity_strategies.ActivityIdealVolume.get_name

*method*

```python
def get_name() -> str
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

Return the type of the activity strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L89-L91)

### particula.particles.activity_strategies.ActivityIdealVolume.neutral_activities

*method*

```python
def neutral_activities(masses, temperature, targets, active) -> NDArray[np.float64]
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

Return dimensionless activity in requested material order.

Version 1 receives full particle-by-material masses (kg), the current
temperature in K and a boolean particle occupancy mask. Existing
strategies retain their full-composition unit-pressure evaluation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L58-L74)

### particula.particles.activity_strategies.ActivityIdealVolume.partial_pressure

*method*

```python
def partial_pressure(pure_vapor_pressure: Union[float, NDArray[np.float64]], mass_concentration: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

Calculate the vapor pressure of species in the particle phase.

**Parameters**

- `pure_vapor_pressure` (`Union[float, NDArray[np.float64]]`) — Pure vapor pressure of the species in Pa.
- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Vapor pressure of the particle in Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L93-L110)

## particula.particles.activity_strategies.ActivityKappaParameter

*class*

```python
class ActivityKappaParameter(ActivityStrategy)
```

Bases: `particula.particles.activity_strategies.ActivityStrategy`

Non-ideal activity strategy using the kappa hygroscopic parameter.

**Attributes**

- `kappa` — Kappa hygroscopic parameters (array or scalar).
- `density` — Densities (array or scalar) in kg/m^3.
- `molar_mass` — Molar masses (array or scalar) in kg/mol.
- `water_index` — Index identifying the water species in arrays.

- activity : Computes non-ideal activity using kappa
  hygroscopicity approach.

**Examples**

```py title="Example Usage"
import particula as par
import numpy as np
strategy = par.particles.ActivityKappaParameter(
    kappa=np.array([0.1, 0.0]),
    density=np.array([1000.0, 1200.0]),
    molar_mass=np.array([0.018, 0.058]),
    water_index=0,
)
a = strategy.activity(np.array([1.0, 2.0]))
# a -> ...
```

> **References**
>
> - Petters, M. D., & Kreidenweis, S. M. (2007). A single parameter
>   representation of hygroscopic growth and cloud condensation
>   nucleus activity. Atmospheric Chemistry and Physics, 7(8),
>   1961-1971. [DOI](https://doi.org/10.5194/acp-7-1961-2007).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L386-L473)

### particula.particles.activity_strategies.ActivityKappaParameter.density

*attribute* · *instance attribute*

```python
density = density if density is not None else np.array([0.0], dtype=np.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L438-L442)

### particula.particles.activity_strategies.ActivityKappaParameter.kappa

*attribute* · *instance attribute*

```python
kappa = kappa if kappa is not None else np.array([0.0], dtype=np.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L435-L437)

### particula.particles.activity_strategies.ActivityKappaParameter.molar_mass

*attribute* · *instance attribute*

```python
molar_mass = molar_mass if molar_mass is not None else np.array([0.0], dtype=np.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L443-L447)

### particula.particles.activity_strategies.ActivityKappaParameter.water_index

*attribute* · *instance attribute*

```python
water_index = water_index
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L448-L448)

### particula.particles.activity_strategies.ActivityKappaParameter.provider_interface_version

*attribute* · *class attribute* · *instance attribute*

```python
provider_interface_version = 1
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L56-L56)

### particula.particles.activity_strategies.ActivityKappaParameter.__init__

*method*

```python
def __init__(kappa: Optional[NDArray[np.float64]] = None, density: Optional[NDArray[np.float64]] = None, molar_mass: Optional[NDArray[np.float64]] = None, water_index: int = 0)
```

Initialize the ActivityKappaParameter strategy.

**Parameters**

- `kappa` (`Optional[NDArray[np.float64]]`) (default: `None`) — Kappa hygroscopic parameters (array or scalar).
- `density` (`Optional[NDArray[np.float64]]`) (default: `None`) — Densities in kg/m^3 (array or scalar).
- `molar_mass` (`Optional[NDArray[np.float64]]`) (default: `None`) — Molar masses in kg/mol (array or scalar).
- `water_index` (`int`) (default: `0`) — Index of the water species.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L420-L448)

### particula.particles.activity_strategies.ActivityKappaParameter.activity

*method*

```python
def activity(mass_concentration: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the activity of a species based on mass concentration.

**Parameters**

- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Activity of the species, unitless.

> **References**
>
> - Petters, M. D., & Kreidenweis, S. M. (2007). A single parameter
>   representation of hygroscopic growth and cloud condensation
>   nucleus activity. Atmospheric Chemistry and Physics, 7(8),
>   1961-1971. [DOI](https://doi.org/10.5194/acp-7-1961-2007).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L450-L473)

### particula.particles.activity_strategies.ActivityKappaParameter.get_name

*method*

```python
def get_name() -> str
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

Return the type of the activity strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L89-L91)

### particula.particles.activity_strategies.ActivityKappaParameter.neutral_activities

*method*

```python
def neutral_activities(masses, temperature, targets, active) -> NDArray[np.float64]
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

Return dimensionless activity in requested material order.

Version 1 receives full particle-by-material masses (kg), the current
temperature in K and a boolean particle occupancy mask. Existing
strategies retain their full-composition unit-pressure evaluation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L58-L74)

### particula.particles.activity_strategies.ActivityKappaParameter.partial_pressure

*method*

```python
def partial_pressure(pure_vapor_pressure: Union[float, NDArray[np.float64]], mass_concentration: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

Calculate the vapor pressure of species in the particle phase.

**Parameters**

- `pure_vapor_pressure` (`Union[float, NDArray[np.float64]]`) — Pure vapor pressure of the species in Pa.
- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Vapor pressure of the particle in Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L93-L110)

## particula.particles.activity_strategies.ActivityNonIdealBinary

*class*

```python
class ActivityNonIdealBinary(ActivityStrategy)
```

Bases: `particula.particles.activity_strategies.ActivityStrategy`

Non-ideal activity for binary organic-water mixtures using BAT model.

Uses the Binary Activity Thermodynamics model (Gorkowski et al., 2019)
to compute organic activity coefficients for organic-water mixtures
given mass concentrations. Assumes the last dimension of input is
ordered as [water, organic].

**Parameters**

- `molar_mass` (`float`) — Organic species molar mass in kg/mol.
- `oxygen2carbon` (`float`) — Oxygen to carbon atomic ratio (dimensionless).
- `density` (`float`) — Organic species density in kg/m^3.
- `functional_group` (`Optional[Union[str, List[str]]]`) (default: `None`) — Optional functional group identifier for OH- equivalent conversion in the BAT helper.

**Examples**

```pycon
>>> import numpy as np
>>> from particula.particles.activity_strategies import (
...     ActivityNonIdealBinary,
... )
>>> strategy = ActivityNonIdealBinary(
...     molar_mass=0.2,
...     oxygen2carbon=0.5,
...     density=1400.0,
...     functional_group="carboxylic_acid",
... )
>>> mass_concentration = np.array([0.5, 0.5])
>>> activity = strategy.activity(mass_concentration)
>>> float(activity)
0.0...
```

> **References**
>
> Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
> Relative-humidity-dependent organic aerosol thermodynamics via an
> efficient reduced-complexity model. Atmospheric Chemistry and
> Physics, 19(19), 13383-13410. https://doi.org/10.5194/acp-19-13383-2019

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L242-L352)

### particula.particles.activity_strategies.ActivityNonIdealBinary.density

*attribute* · *instance attribute*

```python
density = density
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L291-L291)

### particula.particles.activity_strategies.ActivityNonIdealBinary.functional_group

*attribute* · *instance attribute*

```python
functional_group = functional_group
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L292-L292)

### particula.particles.activity_strategies.ActivityNonIdealBinary.molar_mass_g

*attribute* · *instance attribute*

```python
molar_mass_g = self.molar_mass_kg * 1000.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L289-L289)

### particula.particles.activity_strategies.ActivityNonIdealBinary.molar_mass_kg

*attribute* · *instance attribute*

```python
molar_mass_kg = float(molar_mass)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L288-L288)

### particula.particles.activity_strategies.ActivityNonIdealBinary.oxygen2carbon

*attribute* · *instance attribute*

```python
oxygen2carbon = oxygen2carbon
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L290-L290)

### particula.particles.activity_strategies.ActivityNonIdealBinary.provider_interface_version

*attribute* · *class attribute* · *instance attribute*

```python
provider_interface_version = 1
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L56-L56)

### particula.particles.activity_strategies.ActivityNonIdealBinary.__init__

*method*

```python
def __init__(molar_mass: float, oxygen2carbon: float, density: float, functional_group: Optional[Union[str, List[str]]] = None) -> None
```

Initialize the binary non-ideal activity strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L280-L292)

### particula.particles.activity_strategies.ActivityNonIdealBinary._activity_pair

*method*

```python
def _activity_pair(mass_concentration)
```

Evaluate water and organic activities at the same composition.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L312-L348)

### particula.particles.activity_strategies.ActivityNonIdealBinary.activity

*method*

```python
def activity(mass_concentration: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate organic activity using the BAT model.

**Parameters**

- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Binary mass concentrations with last dimension size 2 ordered as [water, organic], in kg/m^3.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Organic activity (dimensionless) matching the BAT helper.

**Raises**

- `ValueError` — If the input is scalar or last dimension is not 2.
- `ValueError` — If any total moles are non-positive.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L294-L310)

### particula.particles.activity_strategies.ActivityNonIdealBinary.get_name

*method*

```python
def get_name() -> str
```

Return the strategy identifier.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L350-L352)

### particula.particles.activity_strategies.ActivityNonIdealBinary.neutral_activities

*method*

```python
def neutral_activities(masses, temperature, targets, active) -> NDArray[np.float64]
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

Return dimensionless activity in requested material order.

Version 1 receives full particle-by-material masses (kg), the current
temperature in K and a boolean particle occupancy mask. Existing
strategies retain their full-composition unit-pressure evaluation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L58-L74)

### particula.particles.activity_strategies.ActivityNonIdealBinary.partial_pressure

*method*

```python
def partial_pressure(pure_vapor_pressure: Union[float, NDArray[np.float64]], mass_concentration: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.particles.activity_strategies.ActivityStrategy`

Calculate the vapor pressure of species in the particle phase.

**Parameters**

- `pure_vapor_pressure` (`Union[float, NDArray[np.float64]]`) — Pure vapor pressure of the species in Pa.
- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Vapor pressure of the particle in Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L93-L110)

## particula.particles.activity_strategies.ActivityStrategy

*class*

```python
class ActivityStrategy(ABC)
```

Bases: `ABC`

Abstract base class for vapor pressure and activity calculations.

This interface is used by subclasses for computing particle activity
and partial pressures. General methods include activity() and
partial_pressure().

Methods:
- get_name : Return the type of the activity strategy.
- activity : Calculate the activity of a species. (abstract method)
- partial_pressure : Calculate the partial pressure of a species
    using its pure vapor pressure and computed activity.

**Examples**

```py title="Example Subclass"
class CustomActivity(ActivityStrategy):
    def activity(self, mass_concentration):
        return 1.0

my_activity = CustomActivity()
pvap = my_activity.partial_pressure(101325.0, 1.0)
# pvap -> 101325.0
```

References:
- "Vapor Pressure,"
    [Wikipedia](https://en.wikipedia.org/wiki/Vapor_pressure).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L27-L110)

### particula.particles.activity_strategies.ActivityStrategy.provider_interface_version

*attribute* · *class attribute* · *instance attribute*

```python
provider_interface_version = 1
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L56-L56)

### particula.particles.activity_strategies.ActivityStrategy.activity

*method* · *abstract*

```python
def activity(mass_concentration: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the activity of a species based on its mass concentration.

**Parameters**

- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Activity of the species, unitless.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L76-L87)

### particula.particles.activity_strategies.ActivityStrategy.get_name

*method*

```python
def get_name() -> str
```

Return the type of the activity strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L89-L91)

### particula.particles.activity_strategies.ActivityStrategy.neutral_activities

*method*

```python
def neutral_activities(masses, temperature, targets, active) -> NDArray[np.float64]
```

Return dimensionless activity in requested material order.

Version 1 receives full particle-by-material masses (kg), the current
temperature in K and a boolean particle occupancy mask. Existing
strategies retain their full-composition unit-pressure evaluation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L58-L74)

### particula.particles.activity_strategies.ActivityStrategy.partial_pressure

*method*

```python
def partial_pressure(pure_vapor_pressure: Union[float, NDArray[np.float64]], mass_concentration: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the vapor pressure of species in the particle phase.

**Parameters**

- `pure_vapor_pressure` (`Union[float, NDArray[np.float64]]`) — Pure vapor pressure of the species in Pa.
- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the species in kg/m^3.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Vapor pressure of the particle in Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_strategies.py#L93-L110)
