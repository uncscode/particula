# Brownian Kernel

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Brownian Kernel

> Auto-generated documentation for [particula.next.dynamics.coagulation.brownian_kernel](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/brownian_kernel.py) module.

## brownian_coagulation_kernel

[Show source in brownian_kernel.py:109](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/brownian_kernel.py#L109)

Returns the Brownian coagulation kernel for aerosol particles. Defined
as the product of the diffusivity of the particles, the collection term
`g`, and the radius of the particles.

Args
----
radius_particle : The radius of the particles [m].
diffusivity_particle : The diffusivity of the particles [m^2/s].
g_collection_term_particle : The collection term for Brownian coagulation
[dimensionless].
alpha_collision_efficiency : The collision efficiency of the particles
[dimensionless].

Returns
-------
Square matrix of Brownian coagulation kernel for aerosol particles [m^3/s].

References
----------
Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
Coefficient K12 (with alpha collision efficiency term 13.56)

#### Signature

```python
def brownian_coagulation_kernel(
    radius_particle: Union[float, NDArray[np.float64]],
    diffusivity_particle: Union[float, NDArray[np.float64]],
    g_collection_term_particle: Union[float, NDArray[np.float64]],
    mean_thermal_speed_particle: Union[float, NDArray[np.float64]],
    alpha_collision_efficiency: Union[float, NDArray[np.float64]] = 1.0,
) -> Union[float, NDArray[np.float64]]: ...
```



## brownian_coagulation_kernel_via_system_state

[Show source in brownian_kernel.py:178](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/brownian_kernel.py#L178)

Returns the Brownian coagulation kernel for aerosol particles,
calculating the intermediate properties needed.

#### Arguments

radius_particle : The radius of the particles [m].
mass_particle : The mass of the particles [kg].
temperature : The temperature of the air [K].
pressure : The pressure of the air [Pa].
alpha_collision_efficiency : The collision efficiency of the particles
    [dimensionless].

#### Returns

Square matrix of Brownian coagulation kernel for aerosol particles
    [m^3/s].

#### References

Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
Coefficient K12.

#### Signature

```python
def brownian_coagulation_kernel_via_system_state(
    radius_particle: Union[float, NDArray[np.float64]],
    mass_particle: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    alpha_collision_efficiency: Union[float, NDArray[np.float64]] = 1.0,
) -> Union[float, NDArray[np.float64]]: ...
```



## brownian_diffusivity

[Show source in brownian_kernel.py:83](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/brownian_kernel.py#L83)

Returns the diffusivity of the particles due to Brownian motion

THis is just the scaled aerodynamic mobility of the particles.

Args
----
- temperature : The temperature of the air [K].
- aerodynamic_mobility : The aerodynamic mobility of the particles [m^2/s].

Returns
-------
The diffusivity of the particles due to Brownian motion [m^2/s].

References
----------
Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
Coefficient K12

#### Signature

```python
def brownian_diffusivity(
    temperature: Union[float, NDArray[np.float64]],
    aerodynamic_mobility: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## g_collection_term

[Show source in brownian_kernel.py:49](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/brownian_kernel.py#L49)

Returns the `g` collection term for Brownian coagulation.

Defined as the ratio of the mean free path of the particles to the
radius of the particles.

Args
----
mean_free_path_particle : The mean free path of the particles [m].
radius_particle : The radius of the particles [m].

Returns
-------
The collection term for Brownian coagulation [dimensionless].

References
----------
Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
Coefficient K12

The np.sqrt(2) term appears to be an error in the text, as the term is
not used in the second edition of the book. And when it it is used, the
values are too small, by about 2x.

#### Signature

```python
def g_collection_term(
    mean_free_path_particle: Union[float, NDArray[np.float64]],
    radius_particle: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## mean_free_path_l

[Show source in brownian_kernel.py:18](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/brownian_kernel.py#L18)

Calculate the mean free path of particles for coagulation.

Calculate the mean free path of particles, defined for Brownian
coagulation as the ratio of the diffusivity of the particles to their mean
thermal speed. This parameter is crucial for understanding particle
dynamics in a fluid.

#### Arguments

----
- diffusivity_particle : The diffusivity of the particles [m^2/s].
- mean_thermal_speed_particle : The mean thermal speed of the particles
[m/s].

#### Returns

-------
The mean free path of the particles [m].

#### References

----------
Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
Coefficient K12.

#### Signature

```python
def mean_free_path_l(
    diffusivity_particle: Union[float, NDArray[np.float64]],
    mean_thermal_speed_particle: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```
