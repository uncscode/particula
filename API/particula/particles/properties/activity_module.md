# particula.particles.properties.activity_module

*module*

Activity Functions Module.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/activity_module.py#L1-L1)

## particula.particles.properties.activity_module.get_ideal_activity_mass

*function*

```python
def get_ideal_activity_mass(mass_concentration: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute the ideal activity based on mass fractions.

This function calculates the activity of a species using its mass fraction.
In an ideal mixture, the activity (aᵢ) can be expressed as:

- aᵢ = wᵢ
    - wᵢ is the mass fraction of species i.

**Parameters**

- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Mass concentration of the species in kg/m³.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Ideal activity of the species as a dimensionless value.

**Examples**

``` py title="Example"
import particula as par
par.particles.get_ideal_activity_mass(np.array([1.0, 2.0]))
# Output: array([...])
```

> **References**
>
> - Raoult's Law, "Raoult's law," Wikipedia,
>   https://en.wikipedia.org/wiki/Raoult%27s_law.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/activity_module.py#L110-L147)

## particula.particles.properties.activity_module.get_ideal_activity_molar

*function*

```python
def get_ideal_activity_molar(mass_concentration: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute the ideal activity based on mole fractions.

This function calculates the activity of a species using its mole fraction,
which follows Raoult's Law. The ideal activity (aᵢ) is determined using:

- aᵢ = Xᵢ
    - Xᵢ is the mole fraction of species i.

**Parameters**

- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Mass concentration of the species in kg/m³.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the species in kg/mol.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Ideal activity of the species as a dimensionless value.

**Examples**

``` py title="Example"
import particula as par
par.particles.get_ideal_activity_molar(
    mass_concentration=np.array([1.0, 2.0]),
    molar_mass=np.array([18.015, 28.97])
)
# Output: array([...])
```

> **References**
>
> - Raoult's Law, "Raoult's law," Wikipedia,
>   https://en.wikipedia.org/wiki/Raoult%27s_law.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/activity_module.py#L16-L60)

## particula.particles.properties.activity_module.get_ideal_activity_volume

*function*

```python
def get_ideal_activity_volume(mass_concentration: Union[float, NDArray[np.float64]], density: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute the ideal activity based on volume fractions.

This function calculates the activity of a species using its volume
fraction. In an ideal mixture, the activity (aᵢ) can be expressed as:

- aᵢ = φᵢ
    - φᵢ is the volume fraction of species i.

**Parameters**

- `mass_concentration` (`Union[float, NDArray[np.float64]]`) — Mass concentration of the species in kg/m³.
- `density` (`Union[float, NDArray[np.float64]]`) — Density of the species in kg/m³.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Ideal activity of the species as a dimensionless value.

**Examples**

``` py title="Example"
import particula as par
par.particles.get_ideal_activity_volume(
    mass_concentration=np.array([1.0, 2.0]),
    density=np.array([1000.0, 1200.0])
)
# Output: array([...])
```

> **References**
>
> - Raoult's Law, "Raoult's law," Wikipedia,
>   https://en.wikipedia.org/wiki/Raoult%27s_law.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/activity_module.py#L63-L107)

## particula.particles.properties.activity_module.get_kappa_activity

*function*

```python
def get_kappa_activity(mass_concentration: NDArray[np.float64], kappa: NDArray[np.float64], density: NDArray[np.float64], molar_mass: NDArray[np.float64], water_index: int) -> NDArray[np.float64]
```

Compute species activity using κ hygroscopic growth parameter.

This function calculates the activity of a mixture by combining
volume-fraction weighted κ-values. The water activity (aₘ) is
determined by:

- aₘ = 1 / (1 + κₑ ( Vₛ / Vₐ ))
    - κₑ is the volume-fraction weighted hygroscopic parameter.
    - Vₛ is the total solute volume fraction (all species except water).
    - Vₐ is the water volume fraction.

**Parameters**

- `mass_concentration` (`NDArray[np.float64]`) — Array of mass concentrations in kg/m³.
- `kappa` (`NDArray[np.float64]`) — Array of κ (kappa) hygroscopic parameters, dimensionless.
- `density` (`NDArray[np.float64]`) — Array of densities in kg/m³ for each species.
- `molar_mass` (`NDArray[np.float64]`) — Array of molar masses in kg/mol for each species.
- `water_index` (`int`) — Index of the water component in the arrays.

**Returns**

- (`NDArray[np.float64]`) — Array of species activities, dimensionless.

**Examples**

``` py title="Example"
import particula as par
par.particles.get_kappa_activity(
    mass_concentration=np.array([[1.0, 2.0], [3.0, 4.0]]),
    kappa=np.array([0.0, 0.2]),
    density=np.array([1000.0, 1200.0]),
    molar_mass=np.array([18.015, 28.97]),
    water_index=0
)
# Output: array([...])
```

> **References**
>
> - Petters, M. D., & Kreidenweis, S. M. (2007). "A single parameter
>   representation of hygroscopic growth and cloud condensation nucleus
>   activity," Atmospheric Chemistry and Physics, 7(8), 1961-1971.
>   DOI: https://doi.org/10.5194/acp-7-1961-2007.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/activity_module.py#L150-L258)

## particula.particles.properties.activity_module.get_surface_partial_pressure

*function*

```python
def get_surface_partial_pressure(pure_vapor_pressure: Union[float, NDArray[np.float64]], activity: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute the partial pressure from activity and pure vapor pressure.

This function calculates the partial pressure (pᵢ) of a species, given its
activity (aᵢ) and pure vapor pressure (pᵢ*). It follows:

- pᵢ = aᵢ × pᵢ*

**Parameters**

- `pure_vapor_pressure` (`Union[float, NDArray[np.float64]]`) — Pure vapor pressure of the species in Pa.
- `activity` (`Union[float, NDArray[np.float64]]`) — Activity of the species, dimensionless.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Partial pressure of the species in Pa.

**Examples**

``` py title="Example"
import particula as par
par.particles.get_surface_partial_pressure(1000.0, 0.95)
# Output: 950.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/activity_module.py#L261-L292)
