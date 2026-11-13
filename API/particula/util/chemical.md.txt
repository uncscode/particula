# particula.util.chemical

*module*

Chemical utilities for vapor pressure, surface tension, and search.
Wraps around the `thermo` package for chemical properties.

This module may move up in the package in the future, if it becomes
more widely used.

[View source](https://github.com/uncscode/particula/blob/main/particula/util/chemical/__init__.py#L1-L1)

## particula.util.chemical.get_chemical_search

*function*

```python
def get_chemical_search(identifier: str) -> str
```

Re-exported from: `particula.util.chemical.chemical_search`

Resolve a chemical identifier to a canonical name or CAS number.

The resolution proceeds in two stages:

1. Exact search through ``chemicals.identifiers.CAS_from_any``
   (returns immediately when successful).
2. Case-insensitive fuzzy matching against the PubChem name index that
   ships with *thermo* (Levenshtein distance via ``difflib``).

**Parameters**

- `identifier` (`str`) — Arbitrary chemical name, formula, InChI, or CAS registry number.

**Returns**

- (`str`) — The resolved identifier string when a match is found, otherwise ``None``.

**Examples**

``` py title="Exact CAS lookup"
from particula.util.materials.chemical_search import get_chemical_search
assert get_chemical_search("64-17-5") == "64-17-5"      # ethanol
```
``` py title="Fuzzy text match"
hit = get_chemical_search("soodim chlorid")             # typo
# hit -> "7647-14-5"  (sodium chloride)
```

> **References**
>
> - Wilson, N., et al., "Open-source tools for chemical identifiers,"
>   *J. Chem. Inf. Model.* **60** (2020) 833–839.
> - "Chemical identifier,"
>   [Wikipedia](https://en.wikipedia.org/wiki/Chemical_identifier).

[View source](https://github.com/uncscode/particula/blob/main/particula/util/chemical/chemical_search.py#L24-L83)

## particula.util.chemical.get_chemical_stp_properties

*function*

```python
def get_chemical_stp_properties(identifier: str) -> Dict[str, float]
```

Re-exported from: `particula.util.chemical.chemical_properties`

Return selected thermodynamic properties of *identifier* at STP.

Parameters
----------
identifier
    Any string accepted by ``thermo.Chemical`` (name, formula, CAS).

Returns:
-------
dict
    {
        "molar_mass": kg mol⁻¹,
        "density": kg m⁻³,
        "surface_tension": N m⁻¹,
        "pure_vapor_pressure": Pa,
    }

[View source](https://github.com/uncscode/particula/blob/main/particula/util/chemical/chemical_properties.py#L8-L46)

## particula.util.chemical.get_chemical_surface_tension

*function*

```python
def get_chemical_surface_tension(chemical_identifier: str, temperature: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]
```

Re-exported from: `particula.util.chemical.chemical_surface_tension`

Calculate the surface tension of a pure chemical in N/m.

The calculation delegates to ``thermo.chemical.Chemical.SurfaceTension``,
evaluated at the requested temperature(s):

- σ = σ(T)
    - σ is the surface tension in newtons per metre (N/m),
    - T is the absolute temperature in kelvin (K).

**Parameters**

- `chemical_identifier` (`str`) — Identifier accepted by ``thermo.chemical.Chemical`` (name, CAS number, or formula).
- `temperature` (`Union[float, NDArray[np.float64]]`) — Scalar or array of temperatures in kelvin (K).

**Returns**

- (`NDArray[np.float64]`) — Surface tension(s) in N/m with the same shape as ``temperature``. A scalar input returns a 0-d ``numpy.ndarray`` for consistency.

**Examples**

``` py title="Scalar temperature"
from particula.util.materials.surface_tension import             get_surface_tension
get_surface_tension("water", 298.15)
# Output: 0.072 (≈ N/m)
```

``` py title="Array temperature"
import numpy as np
from particula.util.materials.surface_tension import             get_surface_tension
temps = np.array([280.0, 298.15, 320.0])
get_surface_tension("water", temps)
# Output: array([0.076, 0.072, 0.065])
```

> **References**
>
> - "Surface tension",
>   [Wikipedia](https://en.wikipedia.org/wiki/Surface_tension)
> - D. R. Lide, *CRC Handbook of Chemistry and Physics*, 90th ed.,
>   CRC Press, 2009.

[View source](https://github.com/uncscode/particula/blob/main/particula/util/chemical/chemical_surface_tension.py#L11-L67)

## particula.util.chemical.get_chemical_vapor_pressure

*function*

```python
def get_chemical_vapor_pressure(chemical_identifier: str, temperature: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]
```

Re-exported from: `particula.util.chemical.chemical_vapor_pressure`

Retrieve the saturation-vapor pressure of a chemical at temperature.

The saturation-vapor pressure (Pₛₐₜ) is obtained by calling the correlation
implemented in ``thermo.chemical.Chemical``:

- Pₛₐₜ = Chemical(chemical_identifier).VaporPressure(T)
    - Pₛₐₜ is saturation-vapor pressure in Pascals (Pa),
    - T is temperature in Kelvin.

**Parameters**

- `chemical_identifier` (`str`) — Identifier accepted by ``thermo.chemical.Chemical`` (name, CAS number, or formula).
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature(s) in Kelvin. May be a scalar or a NumPy array.

**Returns**

- (`NDArray[np.float64]`) — Saturation-vapor pressure(s) in Pascals (Pa). For a scalar temperature, a 0-d NumPy array is returned for consistency.

**Examples**

``` py title="Example Usage – scalar input"
from particula.util.materials.vapor_pressure import get_vapor_pressure
p_sat = get_vapor_pressure("water", 298.15)
# p_sat ≈ 3169.0  # Pa
```

``` py title="Example Usage – vectorised input"
import numpy as np
from particula.util.materials.vapor_pressure import get_vapor_pressure
temps = np.array([280.0, 290.0, 300.0])
p_sat = get_vapor_pressure("water", temps)
# p_sat → array([...])  # Pa
```

> **References**
>
> - "Vapour pressure,"
>   [Wikipedia](https://en.wikipedia.org/wiki/Vapour_pressure)
> - R. H. Perry & D. W. Green, *Perry's Chemical Engineers' Handbook*,
>   8th ed., McGraw-Hill, 2007.

[View source](https://github.com/uncscode/particula/blob/main/particula/util/chemical/chemical_vapor_pressure.py#L11-L66)

**Modules**

- `particula.util.chemical.chemical_properties`
- `particula.util.chemical.chemical_search`
- `particula.util.chemical.chemical_surface_tension`
- `particula.util.chemical.chemical_vapor_pressure`
- `particula.util.chemical.thermo_import`
