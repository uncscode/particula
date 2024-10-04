# Slip Correction Module

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Slip Correction Module

> Auto-generated documentation for [particula.next.particles.properties.slip_correction_module](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/slip_correction_module.py) module.

## cunningham_slip_correction

[Show source in slip_correction_module.py:9](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/slip_correction_module.py#L9)

Calculate the Cunningham slip correction factor. Accounts for
non-continuum effects on small particles.

#### Arguments

-----
- `-` *knudsen_number* - Knudsen number [unitless].

#### Returns

--------
- Slip correction factor [unitless].

Reference:
----------
- Dimensionless quantity accounting for non-continuum effects
on small particles. It is a deviation from Stokes' Law.
Stokes assumes a no-slip condition that is not correct at
high Knudsen numbers. The slip correction factor is used to
calculate the friction factor.
Thus, the slip correction factor is about unity (1) for larger
particles (Kn -> 0). Its behavior on the other end of the
spectrum (smaller particles; Kn -> inf) is more nuanced, though
it tends to scale linearly on a log-log scale, log Cc vs log Kn.
- https://en.wikipedia.org/wiki/Cunningham_correction_factor

#### Signature

```python
def cunningham_slip_correction(
    knudsen_number: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```
