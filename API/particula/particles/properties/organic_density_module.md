# particula.particles.properties.organic_density_module

*module*

Organic density estimation utilities.

Provides helper functions that implement the Girolami (1994) “back-of-the-
envelope” method for estimating the density of organic compounds from their
elemental composition.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/organic_density_module.py#L1-L1)

## particula.particles.properties.organic_density_module.MASS_C

*attribute* · *module attribute*

```python
MASS_C = 12.01
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/organic_density_module.py#L16-L16)

## particula.particles.properties.organic_density_module.MASS_H

*attribute* · *module attribute*

```python
MASS_H = 1.008
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/organic_density_module.py#L18-L18)

## particula.particles.properties.organic_density_module.MASS_N

*attribute* · *module attribute*

```python
MASS_N = 14.0067
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/organic_density_module.py#L19-L19)

## particula.particles.properties.organic_density_module.MASS_O

*attribute* · *module attribute*

```python
MASS_O = 16.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/organic_density_module.py#L17-L17)

## particula.particles.properties.organic_density_module.get_organic_density_array

*function*

```python
def get_organic_density_array(molar_mass: Union[list[float], NDArray[np.float64]], oxygen2carbon: Union[list[float], NDArray[np.float64]], hydrogen2carbon: Optional[Union[list[float], NDArray[np.float64]]] = None, nitrogen2carbon: Optional[Union[list[float], NDArray[np.float64]]] = None, mass_ratio_convert: bool = False) -> NDArray[np.float64]
```

Vectorised wrapper around ``get_organic_density_estimate``.

Applies the Girolami density estimate element-wise to one-dimensional NumPy
arrays or lists of molecular properties.

**Parameters**

- `molar_mass` (`Union[list[float], NDArray[np.float64]]`) — Sequence of molar masses (or mass ratios if ``mass_ratio_convert`` is ``True``) in g mol⁻¹.
- `oxygen2carbon` (`Union[list[float], NDArray[np.float64]]`) — Sequence of atomic O:C ratios.
- `hydrogen2carbon` (`Optional[Union[list[float], NDArray[np.float64]]]`) (default: `None`) — Sequence of atomic H:C ratios (may be ``None`` or negative to trigger estimation inside the helper function).
- `nitrogen2carbon` (`Optional[Union[list[float], NDArray[np.float64]]]`) (default: `None`) — Sequence of atomic N:C ratios (optional).
- `mass_ratio_convert` (`bool`) (default: `False`) — Propagate conversion flag to the scalar helper.

**Returns**

- (`NDArray[np.float64]`) — NumPy array of estimated densities in kg m⁻³ having the same shape as the input arrays.

**Examples**

```py title="Batch calculation"
import numpy as np
from particula.particles import (
    get_organic_density_array,
)

mm  = np.array([118.09, 204.23])
oc  = np.array([1.0, 0.5])
hc  = np.array([1.333, 1.714])

rho = get_organic_density_array(mm, oc, hc)
print(np.round(rho, 0))  # array([1560., 1210.])
```

References:
- G. S. Girolami, “A Simple ‘Back of the Envelope’ Method for Estimating
    the Densities and Molecular Volumes of Liquids and Solids,” *J. Chem.
    Educ.*, 71 (11), 962 (1994).  DOI:10.1021/ed071p962

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/organic_density_module.py#L112-L180)

## particula.particles.properties.organic_density_module.get_organic_density_estimate

*function*

```python
def get_organic_density_estimate(molar_mass: float, oxygen2carbon: float, hydrogen2carbon: Optional[float] = None, nitrogen2carbon: Optional[float] = None, mass_ratio_convert: bool = False) -> float
```

Estimate the density of an organic molecule via the Girolami method.

The original paper proposes an empirical two-step approach:

1.  Base density (ρ₀) is obtained from
    ρ₀ = M / (5 · n_C · (2 + H:C + 2 · O:C + 2 · N:C)) #[g cm⁻³]

2.  A polar-functional correction is applied:
    ρ = ρ₀ × [1 + min(0.1 · n_C · (O:C + N:C), 0.3)]

    -  M is the molar mass [g mol⁻¹]
    -  n_C is the number of carbon atoms in the molecule
    -  H:C, O:C, N:C are atomic ratios.

**Parameters**

- `molar_mass` (`float`) — Molar mass of the compound in g mol⁻¹. If ``mass_ratio_convert`` is ``True`` this is interpreted as a mass ratio and is internally converted to molar mass.
- `oxygen2carbon` (`float`) — Atomic O:C ratio.
- `hydrogen2carbon` (`Optional[float]`) (default: `None`) — Atomic H:C ratio.  Supply a *negative* value (or ``None``) to assume H:C = 2 − O:C as suggested by Girolami.
- `nitrogen2carbon` (`Optional[float]`) (default: `None`) — Atomic N:C ratio.  ``None`` defaults to 0.
- `mass_ratio_convert` (`bool`) (default: `False`) — If ``True`` convert ``molar_mass`` from a mass ratio to molar mass using ``particula.activity.ratio.from_molar_mass_ratio``.

**Returns**

- (`float`) — Estimated density of the compound in kg m⁻³.

**Examples**

```py title="Single compound"
from particula.particles import (
    get_organic_density_estimate,
)

# Succinic acid (M = 118.09, O:C = 1, H:C = 1.333)
rho = get_organic_density_estimate(118.09, oxygen2carbon=1.0,
                                   hydrogen2carbon=1.333)
print(round(rho, 0))  # 1560
```

References:
- G. S. Girolami, “A Simple ‘Back of the Envelope’ Method for Estimating
    the Densities and Molecular Volumes of Liquids and Solids,” *J. Chem.
    Educ.*, 71 (11), 962 (1994).  DOI:10.1021/ed071p962

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/organic_density_module.py#L22-L109)
