# particula.particles.properties.convert_mole_fraction

*module*

Convert from mole fractions.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_mole_fraction.py#L1-L1)

## particula.particles.properties.convert_mole_fraction.get_mass_fractions_from_moles

*function*

```python
def get_mass_fractions_from_moles(mole_fractions: NDArray[np.float64], molecular_weights: NDArray[np.float64]) -> NDArray[np.float64]
```

Convert mole fractions to mass fractions for N components.

The relationship between mass fraction (wᵢ) and mole fraction (xᵢ) is:

- wᵢ = (xᵢ × Mᵢ) / Σⱼ(xⱼ × Mⱼ)
    - wᵢ is the mass fraction of component i (unitless),
    - xᵢ is the mole fraction of component i (unitless),
    - Mᵢ is the molecular weight of component i (kg/mol),
    - Σⱼ(xⱼ × Mⱼ) is the total mass (per total moles).

**Parameters**

- `mole_fractions` (`NDArray[np.float64]`) — Mole fractions (unitless). Can be 1D or 2D. If 2D, each row is treated as a set of mole fractions for N components.
- `molecular_weights` (`NDArray[np.float64]`) — Molecular weights (kg/mol). Must match the shape of ``mole_fractions`` in the last dimension.

**Returns**

- (`NDArray[np.float64]`) — Mass fractions (unitless). Rows sum to 1 if input is 2D; returns 1D mass fractions if input is 1D.

**Examples**

``` py title="Example 1: 1D"
import numpy as np
import particula as par
x_1d = np.array([0.2, 0.5, 0.3])    # mole fractions
mw_1d = np.array([18.0, 44.0, 28.0])  # molecular weights
par.get_mass_fractions_from_moles(x_1d, mw_1d)
# Output: ([0.379..., 0.620..., 0.0])
```

``` py title="Example 2: 2D"
import numpy as np
import particula as par
x_2d = np.array([
    [0.2, 0.5, 0.3],
    [0.3, 0.3, 0.4]
])
mw_2d = np.array([18.0, 44.0, 28.0])
par.get_mass_fractions_from_moles(x_2d, mw_2d)
```

> **References**
>
> - Wikipedia contributors, "Mass fraction (chemistry)," Wikipedia,
>   https://en.wikipedia.org/wiki/Mass_fraction_(chemistry).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_mole_fraction.py#L9-L93)
