# Condensation

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Dynamics](./index.md#dynamics) / Condensation

> Auto-generated documentation for [particula.next.dynamics.condensation](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py) module.

## CondensationIsothermal

[Show source in condensation.py:306](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L306)

Condensation strategy for isothermal conditions, where the temperature
remains constant. This class implements the mass transfer rate calculation
for condensation of particles based on partial pressures. No Latent heat
of vaporization effect is considered.

#### Signature

```python
class CondensationIsothermal(CondensationStrategy):
    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float_]],
        diffusion_coefficient: Union[float, NDArray[np.float_]] = 2 * 1e-09,
        accommodation_coefficient: Union[float, NDArray[np.float_]] = 1.0,
    ): ...
```

#### See also

- [CondensationStrategy](#condensationstrategy)

### CondensationIsothermal().mass_transfer_rate

[Show source in condensation.py:325](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L325)

#### Signature

```python
def mass_transfer_rate(
    self,
    particle: ParticleRepresentation,
    gas_species: GasSpecies,
    temperature: float,
    pressure: float,
    dynamic_viscosity: Optional[float] = None,
) -> Union[float, NDArray[np.float_]]: ...
```

#### See also

- [GasSpecies](../gas/species.md#gasspecies)
- [ParticleRepresentation](../particles/representation.md#particlerepresentation)



## CondensationStrategy

[Show source in condensation.py:122](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L122)

Abstract class for mass transfer strategies, for condensation or
evaporation of particles. This class should be subclassed to implement
specific mass transfer strategies.

#### Arguments

-----------
- molar_mass (Union[float, NDArray[np.float_]]): The molar mass of the
species [kg/mol]. If a single value is provided, it will be used for all
species.
- diffusion_coefficient (Union[float, NDArray[np.float_]]): The diffusion
coefficient of the species [m^2/s]. If a single value is provided, it will
be used for all species. Default is 2*1e-9 m^2/s for air.
- accommodation_coefficient (Union[float, NDArray[np.float_]]): The mass
accommodation coefficient of the species. If a single value is provided,
it will be used for all species. Default is 1.0.

#### Signature

```python
class CondensationStrategy(ABC):
    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float_]],
        diffusion_coefficient: Union[float, NDArray[np.float_]] = 2 * 1e-09,
        accommodation_coefficient: Union[float, NDArray[np.float_]] = 1.0,
    ): ...
```

### CondensationStrategy().first_order_mass_transport

[Show source in condensation.py:225](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L225)

Calculate the first-order mass transport coefficient, K, for a given
particle based on the diffusion coefficient, radius, and vapor
transition correction factor.

#### Arguments

-----
- radius (Union[float, NDArray[np.float_]]): The radius of the particle
[m].
- temperature (float): The temperature at which the first-order mass
transport coefficient is to be calculated.
- pressure (float): The pressure of the gas phase.
- dynamic_viscosity (Optional[float]): The dynamic viscosity of the gas
[Pa*s]. If not provided, it will be calculated based on the temperature

#### Returns

--------
- Union[float, NDArray[np.float_]]: The first-order mass transport
coefficient per particle (m^3/s).

#### References

----------
- Aerosol Modeling, Chapter 2, Equation 2.49 (excluding particle
number)

#### Signature

```python
def first_order_mass_transport(
    self,
    radius: Union[float, NDArray[np.float_]],
    temperature: float,
    pressure: float,
    dynamic_viscosity: Optional[float] = None,
) -> Union[float, NDArray[np.float_]]: ...
```

### CondensationStrategy().knudsen_number

[Show source in condensation.py:187](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L187)

Calculate the Knudsen number based on the mean free path of the gas
molecules and the radius of the particle.

#### Arguments

-----
- radius (Union[float, NDArray[np.float_]]): The radius of the particle
[m].
- temperature (float): The temperature of the gas [K].
- pressure (float): The pressure of the gas [Pa].
- dynamic_viscosity (Optional[float]): The dynamic viscosity of the gas
[Pa*s]. If not provided, it will be calculated based on the temperature

#### Returns

--------
- Union[float, NDArray[np.float_]]: The Knudsen number, which is the
ratio of the mean free path to the particle radius.

#### References

----------
- https://en.wikipedia.org/wiki/Knudsen_number

#### Signature

```python
def knudsen_number(
    self,
    radius: Union[float, NDArray[np.float_]],
    temperature: float,
    pressure: float,
    dynamic_viscosity: Optional[float] = None,
) -> Union[float, NDArray[np.float_]]: ...
```

### CondensationStrategy().mass_transfer_rate

[Show source in condensation.py:272](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L272)

Calculate the mass transfer rate based on the difference in partial
pressure and the first-order mass transport coefficient.

#### Arguments

-----
- particle (Particle class): The particle for which the mass transfer
rate is to be calculated.
- gas_species (GasSpecies class): The gas species with which the
particle is in contact.
- temperature (float): The temperature at which the mass transfer rate
is to be calculated.
- pressure (float): The pressure of the gas phase.
- dynamic_viscosity (Optional[float]): The dynamic viscosity of the gas
[Pa*s]. If not provided, it will be calculated based on the temperature

#### Returns

--------
- Union[float, NDArray[np.float_]]: The mass transfer rate for the
particle [kg/s].

#### Signature

```python
@abstractmethod
def mass_transfer_rate(
    self,
    particle: ParticleRepresentation,
    gas_species: GasSpecies,
    temperature: float,
    pressure: float,
    dynamic_viscosity: Optional[float] = None,
) -> Union[float, NDArray[np.float_]]: ...
```

#### See also

- [GasSpecies](../gas/species.md#gasspecies)
- [ParticleRepresentation](../particles/representation.md#particlerepresentation)

### CondensationStrategy().mean_free_path

[Show source in condensation.py:152](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L152)

Calculate the mean free path of the gas molecules based on the
temperature, pressure, and dynamic viscosity of the gas.

#### Arguments

-----
- temperature (float): The temperature of the
gas [K].
- pressure (float): The pressure of the gas
[Pa].
- dynamic_viscosity (Optional[float]): The dynamic viscosity of the gas
[Pa*s]. If not provided, it will be calculated based on the temperature

#### Returns

--------
- Union[float, NDArray[np.float_]]: The mean free path of the gas
molecules in meters (m).

#### References

----------
- https://en.wikipedia.org/wiki/Mean_free_path

#### Signature

```python
def mean_free_path(
    self, temperature: float, pressure: float, dynamic_viscosity: Optional[float] = None
) -> Union[float, NDArray[np.float_]]: ...
```



## first_order_mass_transport_k

[Show source in condensation.py:53](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L53)

Calculate the first-order mass transport coefficient, K, for a given radius
diffusion coefficient, and vapor transition correction factor. For a
single particle.

#### Arguments

-----
- radius (Union[float, NDArray[np.float_]]): The radius of the particle
[m].
- diffusion_coefficient (Union[float, NDArray[np.float_]]): The diffusion
coefficient of the vapor [m^2/s], default to air.
- vapor_transition (Union[float, NDArray[np.float_]]): The vapor transition
correction factor. [unitless]

#### Returns

--------
- Union[float, NDArray[np.float_]]: The first-order mass transport
coefficient per particle (m^3/s).

#### References

----------
- Aerosol Modeling, Chapter 2, Equation 2.49 (excluding particle number)
- https://en.wikipedia.org/wiki/Mass_diffusivity

#### Signature

```python
def first_order_mass_transport_k(
    radius: Union[float, NDArray[np.float_]],
    vapor_transition: Union[float, NDArray[np.float_]],
    diffusion_coefficient: Union[float, NDArray[np.float_]] = 2 * 1e-09,
) -> Union[float, NDArray[np.float_]]: ...
```



## mass_transfer_rate

[Show source in condensation.py:85](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/condensation.py#L85)

Calculate the mass transfer rate based on the difference in partial
pressure and the first-order mass transport coefficient.

#### Arguments

-----
- pressure_delta (Union[float, NDArray[np.float_]]): The difference in
partial pressure between the gas phase and the particle phase.
- first_order_mass_transport (Union[float, NDArray[np.float_]]): The
first-order mass transport coefficient per particle.
- temperature (Union[float, NDArray[np.float_]]): The temperature at which
the mass transfer rate is to be calculated.

#### Returns

--------
- Union[float, NDArray[np.float_]]: The mass transfer rate for the particle
[kg/s].

#### References

----------
- Aerosol Modeling, Chapter 2, Equation 2.41 (excluding particle number)
- Seinfeld and Pandis, "Atmospheric Chemistry and Physics", Equation 13.3

#### Signature

```python
def mass_transfer_rate(
    pressure_delta: Union[float, NDArray[np.float_]],
    first_order_mass_transport: Union[float, NDArray[np.float_]],
    temperature: Union[float, NDArray[np.float_]],
    molar_mass: Union[float, NDArray[np.float_]],
) -> Union[float, NDArray[np.float_]]: ...
```
