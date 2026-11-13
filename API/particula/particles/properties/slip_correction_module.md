# particula.particles.properties.slip_correction_module

*module*

Module for calculate slip correction.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/slip_correction_module.py#L1-L1)

## particula.particles.properties.slip_correction_module.get_cunningham_slip_correction

*function*

```python
def get_cunningham_slip_correction(knudsen_number: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the Cunningham slip correction factor for small particles in a
fluid.

The slip correction factor (C_c) accounts for non-continuum effects on
small particles, correcting for the no-slip assumption used in Stokes'
law. It is calculated using:

- C_c = 1 + Kn × (1.257 + 0.4 × exp(-1.1 / Kn))
    - Kn is the dimensionless Knudsen number.

**Parameters**

- `knudsen_number` (`Union[float, NDArray[np.float64]]`) — Knudsen number (dimensionless).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Slip correction factor (dimensionless).

**Examples**

``` py title="Example"
import particula as par
par.particles.get_cunningham_slip_correction(0.1)
# Output: ...
```

> **References**
>
> - "Cunningham correction factor," Wikipedia,
>   https://en.wikipedia.org/wiki/Cunningham_correction_factor

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/slip_correction_module.py#L11-L42)
