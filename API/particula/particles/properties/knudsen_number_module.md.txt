# particula.particles.properties.knudsen_number_module

*module*

Module for calculating Knudsen number.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/knudsen_number_module.py#L1-L1)

## particula.particles.properties.knudsen_number_module.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/knudsen_number_module.py#L11-L11)

## particula.particles.properties.knudsen_number_module.get_knudsen_number

*function*

```python
def get_knudsen_number(mean_free_path: Union[float, NDArray[np.float64]], particle_radius: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate Knudsen number (Kn) from mean free path and radius.

The Knudsen number (Kn) indicates whether a flow is in the continuum
regime or the free molecular regime. It is computed by:

- Kn = λ / r
    - Kn is the Knudsen number (dimensionless),
    - λ is the mean free path in meters (m),
    - r is the particle radius in meters (m).

**Parameters**

- `mean_free_path` (`Union[float, NDArray[np.float64]]`) — Mean free path of the gas molecules in meters (m).
- `particle_radius` (`Union[float, NDArray[np.float64]]`) — Radius of the particle in meters (m).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The Knudsen number, which is the ratio of the mean free path to the particle radius.

**Examples**

``` py title="Example Usage"
import particula as par
par.particles.get_knudsen_number(6.5e-8, 1.0e-7)
# Output: 0.65
```

> **References**
>
> - Knudsen number, Wikipedia,
>   https://en.wikipedia.org/wiki/Knudsen_number

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/knudsen_number_module.py#L14-L88)
