# particula.particles.properties.stokes_number

*module*

Stokes number calculation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/stokes_number.py#L1-L1)

## particula.particles.properties.stokes_number.get_stokes_number

*function*

```python
def get_stokes_number(particle_inertia_time: Union[float, NDArray[np.float64]], kolmogorov_time: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute the Stokes number (St) to measure particle inertia relative to
fluid flow.

The Stokes number is a dimensionless parameter reflecting how much a
particle resists following changes in the fluid’s motion. If St >> 1,
particle inertia dominates; if St << 1, the particle closely follows
fluid flow. Mathematically:

- St = τ_p / τ_k
    - St : Stokes number (dimensionless),
    - τ_p : Particle inertia time [s],
    - τ_k : Kolmogorov timescale [s].

**Parameters**

- `particle_inertia_time` (`Union[float, NDArray[np.float64]]`) — Particle inertia time in seconds (s).
- `kolmogorov_time` (`Union[float, NDArray[np.float64]]`) — Kolmogorov timescale in seconds (s).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Dimensionless Stokes number (float or NDArray[np.float64]).

**Examples**

``` py title="Example"
import particula as par
par.particles.get_stokes_number(1e-3, 2e-3)
# Output: 0.5
```

> **References**
>
> - [Stokes number, Wikipedia](https://en.wikipedia.org/wiki/Stokes_number)
> - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and
>   Physics, 3rd ed., Wiley-Interscience.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/stokes_number.py#L11-L53)
