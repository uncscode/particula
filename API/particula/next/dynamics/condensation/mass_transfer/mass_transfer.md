# Mass Transfer

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Dynamics](../index.md#dynamics) / [Condensation](./index.md#condensation) / Mass Transfer

> Auto-generated documentation for [particula.next.dynamics.condensation.mass_transfer](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/condensation/mass_transfer.py) module.

## calculate_mass_transfer

[Show source in mass_transfer.py:140](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/condensation/mass_transfer.py#L140)

Helper function that routes the mass transfer calculation to either the
single-species or multi-species calculation functions based on the input
dimensions of gas_mass.

#### Arguments

- `mass_rate` - The rate of mass transfer per particle (kg/s).
- `time_step` - The time step for the mass transfer calculation (seconds).
- `gas_mass` - The available mass of gas species (kg).
- `particle_mass` - The mass of each particle (kg).
- `particle_concentration` - The concentration of particles (number/m^3).

#### Returns

The amount of mass transferred, accounting for gas and particle
    limitations.

#### Signature

```python
def calculate_mass_transfer(
    mass_rate: NDArray[np.float64],
    time_step: float,
    gas_mass: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## calculate_mass_transfer_multiple_species

[Show source in mass_transfer.py:225](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/condensation/mass_transfer.py#L225)

Calculate mass transfer for multiple gas species.

#### Arguments

- `mass_rate` - The rate of mass transfer per particle for each gas species
    (kg/s).
- `time_step` - The time step for the mass transfer calculation (seconds).
- `gas_mass` - The available mass of each gas species (kg).
- `particle_mass` - The mass of each particle for each gas species (kg).
- `particle_concentration` - The concentration of particles for each gas
    species (number/m^3).

#### Returns

The amount of mass transferred for multiple gas species.

#### Signature

```python
def calculate_mass_transfer_multiple_species(
    mass_rate: NDArray[np.float64],
    time_step: float,
    gas_mass: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## calculate_mass_transfer_single_species

[Show source in mass_transfer.py:181](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/condensation/mass_transfer.py#L181)

Calculate mass transfer for a single gas species (m=1).

#### Arguments

- `mass_rate` - The rate of mass transfer per particle (number*kg/s).
- `time_step` - The time step for the mass transfer calculation (seconds).
- `gas_mass` - The available mass of gas species (kg).
- `particle_mass` - The mass of each particle (kg).
- `particle_concentration` - The concentration of particles (number/m^3).

#### Returns

The amount of mass transferred for a single gas species.

#### Signature

```python
def calculate_mass_transfer_single_species(
    mass_rate: NDArray[np.float64],
    time_step: float,
    gas_mass: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## first_order_mass_transport_k

[Show source in mass_transfer.py:46](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/condensation/mass_transfer.py#L46)

First-order mass transport coefficient per particle.

Calculate the first-order mass transport coefficient, K, for a given radius
diffusion coefficient, and vapor transition correction factor. For a
single particle.

#### Arguments

- `radius` - The radius of the particle [m].
- `diffusion_coefficient` - The diffusion coefficient of the vapor [m^2/s],
    default to air.
- `vapor_transition` - The vapor transition correction factor. [unitless]

#### Returns

- `Union[float,` *NDArray[np.float64]]* - The first-order mass transport
coefficient per particle (m^3/s).

#### References

- Aerosol Modeling: Chapter 2, Equation 2.49 (excluding number)
- Mass Diffusivity:
    [Wikipedia](https://en.wikipedia.org/wiki/Mass_diffusivity)

#### Signature

```python
def first_order_mass_transport_k(
    radius: Union[float, NDArray[np.float64]],
    vapor_transition: Union[float, NDArray[np.float64]],
    diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-05,
) -> Union[float, NDArray[np.float64]]: ...
```



## mass_transfer_rate

[Show source in mass_transfer.py:83](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/condensation/mass_transfer.py#L83)

Calculate the mass transfer rate for a particle.

Calculate the mass transfer rate based on the difference in partial
pressure and the first-order mass transport coefficient.

#### Arguments

- `pressure_delta` - The difference in partial pressure between the gas
    phase and the particle phase.
- `first_order_mass_transport` - The first-order mass transport coefficient
    per particle.
- `temperature` - The temperature at which the mass transfer rate is to be
    calculated.

#### Returns

The mass transfer rate for the particle [kg/s].

#### References

- Aerosol Modeling Chapter 2, Equation 2.41 (excluding particle number)
- Seinfeld and Pandis: "Atmospheric Chemistry and Physics",
    Equation 13.3

#### Signature

```python
def mass_transfer_rate(
    pressure_delta: Union[float, NDArray[np.float64]],
    first_order_mass_transport: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## radius_transfer_rate

[Show source in mass_transfer.py:117](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/condensation/mass_transfer.py#L117)

Convert mass rate to radius transfer rate.

Convert the mass rate to a radius transfer rate based on the
volume of the particle.

#### Arguments

- `mass_rate` - The mass transfer rate for the particle [kg/s].
- `radius` - The radius of the particle [m].
- `density` - The density of the particle [kg/m^3].

#### Returns

The radius growth rate for the particle [m/s].

#### Signature

```python
def radius_transfer_rate(
    mass_rate: Union[float, NDArray[np.float64]],
    radius: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```
