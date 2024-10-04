# Vapor Correction Module

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Vapor Correction Module

> Auto-generated documentation for [particula.next.particles.properties.vapor_correction_module](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/vapor_correction_module.py) module.

## vapor_transition_correction

[Show source in vapor_correction_module.py:11](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/vapor_correction_module.py#L11)

Calculate the transition correction factor, f(Kn, alpha), for a given
Knudsen number and mass accommodation coefficient. This function is used to
account for the intermediate regime between continuum and free molecular
flow. This is the Suchs and Futugin transition function.

#### Arguments

-----
- knudsen_number (Union[float, NDArray[np.float64]]): The Knudsen number,
which quantifies the relative importance of the mean free path of gas
molecules to the size of the particle.
- mass_accommodation (Union[float, NDArray[np.float64]]): The mass
accommodation coefficient, representing the probability of a gas molecule
sticking to the particle upon collision.

#### Returns

--------
- Union[float, NDArray[np.float64]]: The transition correction value
calculated based on the specified inputs.

#### References

----------
- Seinfeld and Pandis, "Atmospheric Chemistry and Physics", Chapter 12,
equation 12.43.
- `Note` - There are various formulations for this correction, so further
extensions of this function might be necessary depending on specific
requirements.
- Original reference:
FUCHS, N. A., & SUTUGIN, A. G. (1971). HIGH-DISPERSED AEROSOLS.
In Topics in Current Aerosol Research (p. 1). Elsevier.
https://doi.org/10.1016/B978-0-08-016674-2.50006-6

#### Signature

```python
def vapor_transition_correction(
    knudsen_number: Union[float, NDArray[np.float64]],
    mass_accommodation: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```
