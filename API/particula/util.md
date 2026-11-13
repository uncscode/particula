# particula.util

*module*

Import all the utility classes and functions, so they can be accessed from
'from particula import util'

[View source](https://github.com/uncscode/particula/blob/main/particula/util/__init__.py#L1-L1)

## particula.util.TAILWIND

*attribute* · *module attribute*

```python
TAILWIND = {'slate': {'50': '#f8fafc', '100': '#f1f5f9', '200': '#e2e8f0', '300': '#cbd5e1', '400': '#94a3b8', '500': '#64748b', '600': '#475569', '700': '#334155', '800': '#1e293b', '900': '#0f172a'}, 'gray': {'50': '#f9fafb', '100': '#f3f4f6', '200': '#e5e7eb', '300': '#d1d5db', '400': '#9ca3af', '500': '#6b7280', '600': '#4b5563', '700': '#374151', '800': '#1f2937', '900': '#111827'}, 'zinc': {'50': '#fafafa', '100': '#f4f4f5', '200': '#e4e4e7', '300': '#d4d4d8', '400': '#a1a1aa', '500': '#71717a', '600': '#52525b', '700': '#3f3f46', '800': '#27272a', '900': '#18181b'}, 'neutral': {'50': '#fafafa', '100': '#f5f5f5', '200': '#e5e5e5', '300': '#d4d4d4', '400': '#a3a3a3', '500': '#737373', '600': '#525252', '700': '#404040', '800': '#262626', '900': '#171717'}, 'stone': {'50': '#fafaf9', '100': '#f5f5f4', '200': '#e7e5e4', '300': '#d6d3d1', '400': '#a8a29e', '500': '#78716c', '600': '#57534e', '700': '#44403c', '800': '#292524', '900': '#1c1917'}, 'red': {'50': '#fef2f2', '100': '#fee2e2', '200': '#fecaca', '300': '#fca5a5', '400': '#f87171', '500': '#ef4444', '600': '#dc2626', '700': '#b91c1c', '800': '#991b1b', '900': '#7f1d1d'}, 'orange': {'50': '#fff7ed', '100': '#ffedd5', '200': '#fed7aa', '300': '#fdba74', '400': '#fb923c', '500': '#f97316', '600': '#ea580c', '700': '#c2410c', '800': '#9a3412', '900': '#7c2d12'}, 'amber': {'50': '#fffbeb', '100': '#fef3c7', '200': '#fde68a', '300': '#fcd34d', '400': '#fbbf24', '500': '#f59e0b', '600': '#d97706', '700': '#b45309', '800': '#92400e', '900': '#78350f'}, 'yellow': {'50': '#fefce8', '100': '#fef9c3', '200': '#fef08a', '300': '#fde047', '400': '#facc15', '500': '#eab308', '600': '#ca8a04', '700': '#a16207', '800': '#854d0e', '900': '#713f12'}, 'lime': {'50': '#f7fee7', '100': '#ecfccb', '200': '#d9f99d', '300': '#bef264', '400': '#a3e635', '500': '#84cc16', '600': '#65a30d', '700': '#4d7c0f', '800': '#3f6212', '900': '#365314'}, 'green': {'50': '#ecfdf5', '100': '#d1fae5', '200': '#a7f3d0', '300': '#6ee7b7', '400': '#34d399', '500': '#10b981', '600': '#059669', '700': '#047857', '800': '#065f46', '900': '#064e3b'}, 'emerald': {'50': '#ecfdf5', '100': '#d1fae5', '200': '#a7f3d0', '300': '#6ee7b7', '400': '#34d399', '500': '#10b981', '600': '#059669', '700': '#047857', '800': '#065f46', '900': '#064e3b'}, 'teal': {'50': '#f0fdfa', '100': '#ccfbf1', '200': '#99f6e4', '300': '#5eead4', '400': '#2dd4bf', '500': '#14b8a6', '600': '#0d9488', '700': '#0f766e', '800': '#115e59', '900': '#134e4a'}, 'cyan': {'50': '#ecfeff', '100': '#cffafe', '200': '#a5f3fc', '300': '#67e8f9', '400': '#22d3ee', '500': '#06b6d4', '600': '#0891b2', '700': '#0e7490', '800': '#155e75', '900': '#164e63'}, 'sky': {'50': '#f0f9ff', '100': '#e0f2fe', '200': '#bae6fd', '300': '#7dd3fc', '400': '#38bdf8', '500': '#0ea5e9', '600': '#0284c7', '700': '#0369a1', '800': '#075985', '900': '#0c4a6e'}, 'blue': {'50': '#eff6ff', '100': '#dbeafe', '200': '#bfdbfe', '300': '#93c5fd', '400': '#60a5fa', '500': '#3b82f6', '600': '#2563eb', '700': '#1d4ed8', '800': '#1e40af', '900': '#1e3a8a'}, 'indigo': {'50': '#eef2ff', '100': '#e0e7ff', '200': '#c7d2fe', '300': '#a5b4fc', '400': '#818cf8', '500': '#6366f1', '600': '#4f46e5', '700': '#4338ca', '800': '#3730a3', '900': '#312e81'}, 'violet': {'50': '#f5f3ff', '100': '#ede9fe', '200': '#ddd6fe', '300': '#c4b5fd', '400': '#a78bfa', '500': '#8b5cf6', '600': '#7c3aed', '700': '#6d28d9', '800': '#5b21b6', '900': '#4c1d95'}, 'purple': {'50': '#faf5ff', '100': '#f3e8ff', '200': '#e9d5ff', '300': '#d8b4fe', '400': '#c084fc', '500': '#a855f7', '600': '#9333ea', '700': '#7e22ce', '800': '#6b21a8', '900': '#581c87'}, 'fuchsia': {'50': '#fdf4ff', '100': '#fae8ff', '200': '#f5d0fe', '300': '#f0abfc', '400': '#e879f9', '500': '#d946ef', '600': '#c026d3', '700': '#a21caf', '800': '#86198f', '900': '#701a75'}, 'pink': {'50': '#fdf2f8', '100': '#fce7f3', '200': '#fbcfe8', '300': '#f9a8d4', '400': '#f472b6', '500': '#ec4899', '600': '#db2777', '700': '#be185d', '800': '#9d174d', '900': '#831843'}, 'rose': {'50': '#fff1f2', '100': '#ffe4e6', '200': '#fecdd3', '300': '#fda4af', '400': '#fb7185', '500': '#f43f5e', '600': '#e11d48', '700': '#be123c', '800': '#9f1239', '900': '#881337'}}
```

Re-exported from: `particula.util.colors`

[View source](https://github.com/uncscode/particula/blob/main/particula/util/colors.py#L9-L274)

## particula.util.get_arbitrary_round

*function*

```python
def get_arbitrary_round(values: Union[float, list[float], np.ndarray], base: Union[float, np.float64] = 1.0, mode: str = 'round', nonzero_edge: bool = False) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.util.arbitrary_round`

Round values to the nearest multiple of a specified base.

The function supports "round", "floor", or "ceil" modes, and can retain
original nonzero values if rounding returns zero.

**Parameters**

- `values` (`Union[float, list[float], np.ndarray]`) — The values to be rounded.
- `base` (`Union[float, np.float64]`) (default: `1.0`) — Positive float indicating the rounding interval.
- `mode` (`str`) (default: `'round'`) — Rounding mode, one of ['round', 'floor', 'ceil'].
- `nonzero_edge` (`bool`) (default: `False`) — If True, zeros after rounding are replaced with the original values.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The input values rounded according to the specified base and mode.

**Examples**

``` py title="Example Usage"
import numpy as np
import particula as par

arr = np.array([1.2, 2.5, 3.7, 4.0])
print(par.get_arbitrary_round(arr, base=1.0, mode='round'))
# Output: [1.  2.  4.  4.]

print(par.get_arbitrary_round(arr, base=0.5, mode='floor'))
# Output: [1.  2.  3.5 4. ]

print(par.get_arbitrary_round(2.5, base=1.0, mode='round'))
# Output: 2.0
```

> **References**
>
> - "Rounding," Python Documentation, docs.python.org.
> - "NumPy Rounding," NumPy Documentation, NumPy.org.

[View source](https://github.com/uncscode/particula/blob/main/particula/util/arbitrary_round.py#L14-L79)

## particula.util.get_chemical_search

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

## particula.util.get_chemical_stp_properties

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

## particula.util.get_chemical_surface_tension

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

## particula.util.get_chemical_vapor_pressure

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

## particula.util.get_coerced_type

*function*

```python
def get_coerced_type(data, dtype)
```

Re-exported from: `particula.util.convert_dtypes`

Coerce the given data to the specified dtype if it is not already of that
type.

**Parameters**

- `data` (`Any`) — The data to be coerced (any type).
- `dtype` (`type`) — The desired data type, e.g. float, int, or np.ndarray.

**Returns**

- `result` (`Any`) — The data converted to the specified type.

**Raises**

- `ValueError` — If the data cannot be coerced to the desired dtype.

**Examples**

``` py title="Coerce integer to float"
import particula as par
x = par.get_coerced_type(1, float)
print(x)
# 1.0
```

``` py title="Coerce list to numpy array"
import numpy as np
import particula as par
arr = par.get_coerced_type([1, 2, 3], np.ndarray)
print(arr)
# [1 2 3]
```

> **References**
>
> - NumPy Documentation: https://numpy.org/doc/

[View source](https://github.com/uncscode/particula/blob/main/particula/util/convert_dtypes.py#L17-L55)

## particula.util.get_dict_from_list

*function*

```python
def get_dict_from_list(list_of_str: list) -> dict
```

Re-exported from: `particula.util.convert_dtypes`

Convert a list of strings into a dictionary mapping each string to its
index.

**Parameters**

- `list_of_str` (`list`) — A non-empty list of strings.

**Returns**

- (`dict`) — A dict where keys are the strings and values are their indices.

**Raises**

- `TypeError` — If the list is empty or contains non-string items.

**Examples**

``` py title="Convert list of strings to dictionary"
import particula as par

str_list = ["alpha", "beta", "gamma"]
mapping = par.get_dict_from_list(str_list)
print(mapping)
# {'alpha': 0, 'beta': 1, 'gamma': 2}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/util/convert_dtypes.py#L58-L93)

## particula.util.get_effective_refractive_index

*function*

```python
def get_effective_refractive_index(m_zero: Union[float, complex], m_one: Union[float, complex], volume_zero: float, volume_one: float) -> Union[float, complex]
```

Re-exported from: `particula.util.refractive_index_mixing`

Calculate the effective refractive index of a two-solute mixture.

The calculation uses volume-weighted molar refraction, described by:
- r_eff = (v0 / (v0 + v1)) * ((m0 - 1) / (m0 + 2)) +
          (v1 / (v0 + v1)) * ((m1 - 1) / (m1 + 2))
    - r_eff is the effective molar refraction,
    - m0, m1 are the refractive indices of each solute,
    - v0, v1 are the volumes of each solute.

Then the resulting refractive index is:
- n_eff = (2 × r_eff + 1) / (1 - r_eff).

**Parameters**

- `m_zero` (`Union[float, complex]`) — Refractive index of solute 0 (float or complex).
- `m_one` (`Union[float, complex]`) — Refractive index of solute 1 (float or complex).
- `volume_zero` (`float`) — Volume of solute 0.
- `volume_one` (`float`) — Volume of solute 1.

**Returns**

- (`Union[float, complex]`) — Effective refractive index of the mixture (float or complex).

**Examples**

``` py title="Example"
import particula as par
n_mix = par.get_effective_refractive_index(1.33, 1.50, 2.0, 1.0)
print(n_mix)
# Output: ~1.382
```

> **References**
>
> - Y. Liu & P. H. Daum, "Relationship of refractive index to mass
>   density and self-consistency mixing rules for multicomponent
>   mixtures like ambient aerosols," Journal of Aerosol Science,
>   vol. 39(11), pp. 974–986, 2008.
>   DOI: 10.1016/j.jaerosci.2008.06.006
> - Wikipedia contributors, "Refractive index," Wikipedia.

[View source](https://github.com/uncscode/particula/blob/main/particula/util/refractive_index_mixing.py#L10-L61)

## particula.util.get_reduced_self_broadcast

*function*

```python
def get_reduced_self_broadcast(alpha_array: NDArray[np.float64]) -> NDArray[np.float64]
```

Re-exported from: `particula.util.reduced_quantity`

Return a square matrix of pairwise reduced values using a single array.

Each element is calculated by broadcasting the array with its transpose:
- r_ij = (α_i × α_j) / (α_i + α_j),
    - r_ij is the reduced quantity between α_i and α_j.

**Parameters**

- `alpha_array` (`NDArray[np.float64]`) — A 1D array for pairwise reduced value calculations.

**Returns**

- (`NDArray[np.float64]`) — A 2D square matrix of pairwise reduced values.

**Examples**

``` py title="Example"
from particula.util.reduced_quantity import get_reduced_self_broadcast
import numpy as np

arr = np.array([1.0, 2.0, 3.0])
print(get_reduced_self_broadcast(arr))
# Output: [[0.5       0.6666667 0.75     ]
#          [0.6666667 1.        1.2      ]
#          [0.75      1.2       1.5      ]]
```

> **References**
>
> - [Reduced Mass, Wikipedia](https://en.wikipedia.org/wiki/Reduced_mass)

[View source](https://github.com/uncscode/particula/blob/main/particula/util/reduced_quantity.py#L70-L111)

## particula.util.get_reduced_value

*function*

```python
def get_reduced_value(alpha: Union[float, NDArray[np.float64]], beta: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.util.reduced_quantity`

Return the reduced value of two parameters.

The reduced value is computed using:
- r = (α × β) / (α + β),
    - r is the reduced quantity,
    - α, β are the input parameters.

**Parameters**

- `alpha` (`Union[float, NDArray[np.float64]]`) — The first parameter (scalar or array).
- `beta` (`Union[float, NDArray[np.float64]]`) — The second parameter (scalar or array).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The element-wise reduced quantity, zero if (α+β)=0.

**Raises**

- `ValueError` — If arrays have incompatible shapes.

**Examples**

``` py title="Example"
from particula.util.reduced_quantity import get_reduced_value
import numpy as np

print(get_reduced_value(3.0, 6.0))
# Output: 2.0

arrA = np.array([1.0, 2.0, 3.0])
arrB = np.array([2.0, 5.0, 10.0])
print(get_reduced_value(arrA, arrB))
# Output: [0.666..., 1.428..., 2.142...]
```

> **References**
>
> - [Reduced Mass, Wikipedia](https://en.wikipedia.org/wiki/Reduced_mass)

[View source](https://github.com/uncscode/particula/blob/main/particula/util/reduced_quantity.py#L15-L67)

## particula.util.get_safe_exp

*function*

```python
def get_safe_exp(value: ArrayLike) -> np.ndarray
```

Re-exported from: `particula.util.machine_limit`

Compute the exponential of each element in the input array, with overflow
protection.

> **The exponential is calculated using**
>
> - y = exp(x), where x is clipped to avoid exceeding machine limits.

**Parameters**

- `value` (`ArrayLike`) — Array-like of values to exponentiate.

**Returns**

- (`np.ndarray`) — np.ndarray of exponentiated values, with machine-level clipping.

**Examples**

``` py title="Example Usage"
import numpy as np
import particula as par

arr = np.array([0, 10, 1000])
print(par.get_safe_exp(arr))
# Output: [1.00000000e+000 2.20264658e+004 1.79769313e+308]
```

> **References**
>
> - "Floating Point Arithmetic," NumPy Documentation, NumPy.org.

[View source](https://github.com/uncscode/particula/blob/main/particula/util/machine_limit.py#L13-L41)

## particula.util.get_safe_log

*function*

```python
def get_safe_log(value: ArrayLike) -> np.ndarray
```

Re-exported from: `particula.util.machine_limit`

Compute the natural logarithm of each element in the input array, with
underflow protection.

> **The natural log is calculated using**
>
> - y = ln(x), where x is clipped away from zero to maintain positivity.

**Parameters**

- `value` (`ArrayLike`) — Array-like of values for logarithm calculation.

**Returns**

- (`np.ndarray`) — np.ndarray of natural logarithms, with machine-level clipping.

**Examples**

``` py title="Example Usage"
import numpy as np
import particula as par

arr = np.array([1e-320, 1.0, 10.0])
print(get_safe_log(arr))
# Output: [-7.40545337e+02  0.00000000e+00  2.30258509e+00]
```

> **References**
>
> - "Logarithms and Machine Precision," NumPy Documentation, NumPy.org.

[View source](https://github.com/uncscode/particula/blob/main/particula/util/machine_limit.py#L44-L72)

## particula.util.get_safe_log10

*function*

```python
def get_safe_log10(value: ArrayLike) -> np.ndarray
```

Re-exported from: `particula.util.machine_limit`

Compute the base-10 logarithm of each element in the input array, with
underflow protection.

> **The base-10 log is calculated using**
>
> - y = log10(x), x clipped from zero to maintain positivity.

**Parameters**

- `value` (`ArrayLike`) — Array-like of values for base-10 logarithm calculation.

**Returns**

- (`np.ndarray`) — np.ndarray of base-10 logarithms, with machine-level clipping.

**Examples**

``` py title="Example Usage"
import numpy as np
import particula as par

arr = np.array([1e-320, 1.0, 1000.0])
print(par.get_safe_log10(arr))
# Output: [-320.           0.           3.        ]
```

> **References**
>
> - "Logarithms and Machine Precision," NumPy Documentation, NumPy.org.

[View source](https://github.com/uncscode/particula/blob/main/particula/util/machine_limit.py#L75-L103)

## particula.util.get_shape_check

*function*

```python
def get_shape_check(time: np.ndarray, data: np.ndarray, header: list) -> np.ndarray
```

Re-exported from: `particula.util.convert_dtypes`

Validate or reshape data array for compatibility with time array
and header list.

If data is 2D, the function attempts to align the time dimension with one
of the axes. If data is 1D, the header list must have exactly one entry.

**Parameters**

- `time` (`np.ndarray`) — 1D array of time values.
- `data` (`Any`) — 1D or 2D array of data values.
- `header` (`list`) — List of headers corresponding to the data dimensions.

**Returns**

- (`np.ndarray`) — A possibly reshaped data array ensuring alignment with time and header constraints.

**Raises**

- `ValueError` — If the header length does not match the data shape,
or if data is 1D but header has more than one entry.

**Examples**

``` py
import numpy as np
import particula as par
time_array = np.arange(0, 10)
data_2d = np.random.rand(10, 5)
headers = ['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5']
reshaped_data = par.get_shape_check(time_array, data_2d, headers)
print(reshaped_data.shape)
# Should be (10, 5)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/util/convert_dtypes.py#L132-L201)

## particula.util.get_unit_conversion

*function*

```python
def get_unit_conversion(old: str, new: str, value: Optional[Union[float, np.ndarray]] = None) -> ReturnType
```

Re-exported from: `particula.util.convert_units`

Convert a numeric value or unit expression from one unit to another using
Pint.

For simple multiplicative units, if no value is provided, this function
returns the conversion factor. For units with an offset
(e.g., temperatures), or if a value is supplied, a fully converted
numeric value is returned instead.

**Parameters**

- `old` (`str`) — A string representing the current unit (e.g., "m", "degC").
- `new` (`str`) — A string representing the target unit.
- `value` (`Optional[Union[float, np.ndarray]]`) (default: `None`) — An optional numeric value to convert. If omitted, returns the conversion factor between old and new.

**Raises**

- `ImportError` — If Pint is not installed. Install it using:
`pip install pint`.

**Returns**

- (`ReturnType`) — A float or ``numpy.ndarray`` representing either the conversion factor or the fully converted value in the target unit.

**Examples**

``` py title="Example Multi-Unit Conversion"
import particula as par
factor = par.get_unit_conversion("ug/m^3", "kg/m^3")
print(factor)
# 1e-9
```

``` py title="Example Temperature Conversion"
import particula as par
degF = par.get_unit_conversion("degC", "degF", value=25)
print(degF)
# ~77.0
```

> **References**
>
> - Pint documentation: https://pint.readthedocs.io/

[View source](https://github.com/uncscode/particula/blob/main/particula/util/convert_units.py#L33-L99)

## particula.util.get_values_of_dict

*function*

```python
def get_values_of_dict(key_list: List[str], dict_to_check: Dict[str, Any]) -> List[Any]
```

Re-exported from: `particula.util.convert_dtypes`

Retrieve a list of index values from a dictionary for the specified keys.

**Parameters**

- `key_list` (`List[str]`) — The keys to look up in the dictionary.
- `dict_to_check` (`Dict[str, Any]`) — The dictionary from which values are retrieved.

**Returns**

- (`List[Any]`) — A list of values corresponding to the given keys.

**Raises**

- `KeyError` — If any key in key_list is not found in dict_to_check.

**Examples**

``` py
import particula as par
my_dict = {'a': 1, 'b': 2, 'c': 3}
vals = par.get_values_of_dict(['a', 'c'], my_dict)
print(vals)
# [1, 3]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/util/convert_dtypes.py#L96-L129)

**Modules**

- `particula.util.arbitrary_round`
- `particula.util.chemical`
- `particula.util.colors`
- `particula.util.constants`
- `particula.util.convert_dtypes`
- `particula.util.convert_units`
- `particula.util.lf2013_coagulation`
- `particula.util.machine_limit`
- `particula.util.reduced_quantity`
- `particula.util.refractive_index_mixing`
- `particula.util.validate_inputs`
