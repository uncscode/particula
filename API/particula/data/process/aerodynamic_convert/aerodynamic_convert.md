# Aerodynamic Convert

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Data](../index.md#data) / [Process](./index.md#process) / Aerodynamic Convert

> Auto-generated documentation for [particula.data.process.aerodynamic_convert](https://github.com/uncscode/particula/blob/main/particula/data/process/aerodynamic_convert.py) module.

## _cost_aerodynamic_radius

[Show source in aerodynamic_convert.py:21](https://github.com/uncscode/particula/blob/main/particula/data/process/aerodynamic_convert.py#L21)

Optimization cost function to determine the aerodynamic radius of a
particle.

#### Arguments

- `guess_aerodynamic_radius` - The initial guess for the aerodynamic radius.
- `mean_free_path_air` - The mean free path of air molecules.
- `particle_radius` - The known physical radius of the particle.
- `kwargs` - Additional keyword arguments for the optimization.
    - density (float): The density of the particle. Default is
        1500 kg/m^3.
    - reference_density (float): The reference density for the
        aerodynamic radius calculation. Default is 1000 kg/m^3.
    - aerodynamic_shape_factor (float): The aerodynamic shape factor.
        Default is 1.0.

#### Returns

The squared error between the guessed aerodynamic radius and
    the calculated aerodynamic radius.

#### Signature

```python
def _cost_aerodynamic_radius(
    guess_aerodynamic_radius: Union[float, NDArray[np.float64]],
    mean_free_path_air: Union[float, NDArray[np.float64]],
    particle_radius: Union[float, NDArray[np.float64]],
    **kwargs
) -> Union[float, NDArray[np.float64]]: ...
```



## _cost_physical_radius

[Show source in aerodynamic_convert.py:80](https://github.com/uncscode/particula/blob/main/particula/data/process/aerodynamic_convert.py#L80)

Optimization cost function to determine the physical radius of a particle.

#### Arguments

- `guess_physical_radius` - The initial guess for the physical radius.
- `mean_free_path_air` - The mean free path of air molecules.
- `aerodynamic_radius` - The known aerodynamic radius of the particle.
- `kwargs` - Additional keyword arguments for the optimization
    - density (float): The density of the particle. Default is
        1500 kg/m^3.
    - reference_density (float): The reference density for the
        aerodynamic radius calculation. Default is 1000 kg/m^3.
    - aerodynamic_shape_factor (float): The aerodynamic shape factor.
        Default is 1.0.

#### Returns

The squared error between the guessed physical radius and the
calculated aerodynamic radius.

#### Signature

```python
def _cost_physical_radius(
    guess_physical_radius: Union[float, NDArray[np.float64]],
    mean_free_path_air: Union[float, NDArray[np.float64]],
    aerodynamic_radius: Union[float, NDArray[np.float64]],
    **kwargs
) -> Union[float, NDArray[np.float64]]: ...
```



## convert_aerodynamic_to_physical_radius

[Show source in aerodynamic_convert.py:139](https://github.com/uncscode/particula/blob/main/particula/data/process/aerodynamic_convert.py#L139)

Convert aerodynamic radius to physical radius for a particle or an array
of particles.

#### Arguments

- `aerodynamic_radius` - The aerodynamic radius or array of radii to be
    converted.
- `pressure` - The ambient pressure in Pascals.
- `temperature` - The ambient temperature in Kelvin.
- `particle_density` - The density of the particles in kg/m^3.
- `aerodynamic_shape_factor` - The aerodynamic shape factor. Default is 1.0.
- `reference_density` - The reference density for the aerodynamic radius
    in kg/m^3. Default is 1000 kg/m^3.

#### Returns

The physical radius or array of radii corresponding to the aerodynamic
radius/radii.

#### Signature

```python
def convert_aerodynamic_to_physical_radius(
    aerodynamic_radius: Union[float, NDArray[np.float64]],
    pressure: float,
    temperature: float,
    particle_density: float,
    aerodynamic_shape_factor: float = 1.0,
    reference_density: float = 1000.0,
) -> Union[float, NDArray[np.float64]]: ...
```



## convert_physical_to_aerodynamic_radius

[Show source in aerodynamic_convert.py:197](https://github.com/uncscode/particula/blob/main/particula/data/process/aerodynamic_convert.py#L197)

Convert physical radius to aerodynamic radius for a particle or an array
of particles.

#### Arguments

- `physical_radius` - The physical radius or array of radii to be converted.
- `pressure` - The ambient pressure in Pascals.
- `temperature` - The ambient temperature in Kelvin.
- `particle_density` - The density of the particles in kg/m^3.
- `aerodynamic_shape_factor` - The aerodynamic shape factor. Default is 1.0.
- `reference_density` - The reference density for the aerodynamic radius
    in kg/m^3. Default is 1000 kg/m^3.

#### Returns

The aerodynamic radius or array of radii corresponding to the physical
radius/radii.

#### Signature

```python
def convert_physical_to_aerodynamic_radius(
    physical_radius: Union[float, NDArray[np.float64]],
    pressure: float,
    temperature: float,
    particle_density: float,
    aerodynamic_shape_factor: float = 1.0,
    reference_density: float = 1000.0,
) -> Union[float, NDArray[np.float64]]: ...
```
