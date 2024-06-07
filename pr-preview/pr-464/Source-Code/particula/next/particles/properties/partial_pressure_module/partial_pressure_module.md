# Partial Pressure Module

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Partial Pressure Module

> Auto-generated documentation for [particula.next.particles.properties.partial_pressure_module](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/properties/partial_pressure_module.py) module.

## partial_pressure_delta

[Show source in partial_pressure_module.py:9](https://github.com/Gorkowski/particula/blob/main/particula/next/particles/properties/partial_pressure_module.py#L9)

Calculate the difference in partial pressure of a species between the gas
phase and the particle phase, which is used in the calculation of the rate
of change of mass of an aerosol particle.

#### Arguments

-----
- partial_pressure_gas (Union[float, NDArray[np.float_]]): The partial
pressure of the species in the gas phase.
- partial_pressure_particle (Union[float, NDArray[np.float_]]): The partial
pressure of the species in the particle phase.
- kelvin_term (Union[float, NDArray[np.float_]]): Kelvin effect to account
for the curvature of the particle.

#### Returns

--------
- Union[float, NDArray[np.float_]]: The difference in partial pressure
between the gas phase and the particle phase.

#### Signature

```python
def partial_pressure_delta(
    partial_pressure_gas: Union[float, NDArray[np.float_]],
    partial_pressure_particle: Union[float, NDArray[np.float_]],
    kelvin_term: Union[float, NDArray[np.float_]],
) -> Union[float, NDArray[np.float_]]: ...
```
