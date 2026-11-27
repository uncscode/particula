# particula.particles.properties.mixing_state_index

*module*

Dry aerosol mixing-state index from particle-by-species masses.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/mixing_state_index.py#L1-L1)

## particula.particles.properties.mixing_state_index.get_mixing_state_index

*function*

```python
def get_mixing_state_index(species_masses: NDArray[np.float64]) -> float
```

Calculate the mass-weighted Riemer–West mixing-state index χ.

For each particle, let H_i be the Shannon entropy of its species mass
fractions, using natural logarithms and the convention 0 ln 0 = 0.
H_alpha is the particle-mass-weighted mean of H_i. H_gamma is the
entropy of bulk species mass fractions. The index is
expm1(H_alpha) / expm1(H_gamma).

**Parameters**

- `species_masses` (`NDArray[np.float64]`) — Nonnegative dry species masses [kg], shape (particles, species). Rows represent individual particles. For representative particles, multiply each row by its represented count before calling. Zero-mass rows are ignored.

**Returns**

- (`float`) — Dimensionless χ in [0, 1]. Pure particles of different species
- (`float`) — give zero. Identical multispecies compositions give one. Returns
- (`float`) — NaN if fewer than two species carry mass, including an empty
- (`float`) — population, because the bulk diversity then gives no contrast.

**Examples**

Two equal-mass particles with opposite 90/10 compositions:

```py
import numpy as np
import particula as par

masses = 1e-18 * np.array([[9., 1.], [1., 9.]])
chi = par.particles.get_mixing_state_index(masses)  # 0.38414549
```

> **References**
>
> Riemer and West (2013), Quantifying aerosol mixing state with
> entropy and diversity measures. https://doi.org/10.5194/acp-13-11423-2013

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/mixing_state_index.py#L10-L67)
