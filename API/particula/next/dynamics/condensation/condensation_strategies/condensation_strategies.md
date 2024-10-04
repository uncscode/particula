# Condensation Strategies

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Dynamics](../index.md#dynamics) / [Condensation](./index.md#condensation) / Condensation Strategies

> Auto-generated documentation for [particula.next.dynamics.condensation.condensation_strategies](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/condensation/condensation_strategies.py) module.

## CondensationIsothermal

[Show source in condensation_strategies.py:288](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/condensation/condensation_strategies.py#L288)

Condensation strategy for isothermal conditions.

Condensation strategy for isothermal conditions, where the temperature
remains constant. This class implements the mass transfer rate calculation
for condensation of particles based on partial pressures. No Latent heat
of vaporization effect is considered.

#### Signature

```python
class CondensationIsothermal(CondensationStrategy):
    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-05,
        accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0,
        update_gases: bool = True,
    ): ...
```

#### See also

- [CondensationStrategy](#condensationstrategy)

### CondensationIsothermal().mass_transfer_rate

[Show source in condensation_strategies.py:311](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/condensation/condensation_strategies.py#L311)

#### Signature

```python
def mass_transfer_rate(
    self,
    particle: ParticleRepresentation,
    gas_species: GasSpecies,
    temperature: float,
    pressure: float,
    dynamic_viscosity: Optional[float] = None,
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [GasSpecies](../../gas/species.md#gasspecies)
- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CondensationIsothermal().rate

[Show source in condensation_strategies.py:363](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/condensation/condensation_strategies.py#L363)

#### Signature

```python
def rate(
    self,
    particle: ParticleRepresentation,
    gas_species: GasSpecies,
    temperature: float,
    pressure: float,
) -> NDArray[np.float64]: ...
```

#### See also

- [GasSpecies](../../gas/species.md#gasspecies)
- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CondensationIsothermal().step

[Show source in condensation_strategies.py:392](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/condensation/condensation_strategies.py#L392)

#### Signature

```python
def step(
    self,
    particle: ParticleRepresentation,
    gas_species: GasSpecies,
    temperature: float,
    pressure: float,
    time_step: float,
) -> Tuple[ParticleRepresentation, GasSpecies]: ...
```

#### See also

- [GasSpecies](../../gas/species.md#gasspecies)
- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)



## CondensationStrategy

[Show source in condensation_strategies.py:60](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/condensation/condensation_strategies.py#L60)

Abstract base class for condensation strategies.

This class defines the interface for various condensation models
used in atmospheric physics. Subclasses should implement specific
condensation algorithms based on different physical models and equations.

#### Arguments

- `molar_mass` - The molar mass of the species [kg/mol]. If a single value
    is provided, it will be used for all species.
- `diffusion_coefficient` - The diffusion coefficient of the species
    [m^2/s]. If a single value is provided, it will be used for all
    species. Default is 2e-5 m^2/s for air.
- `accommodation_coefficient` - The mass accommodation coefficient of the
    species. If a single value is provided, it will be used for all
    species. Default is 1.0.

#### Signature

```python
class CondensationStrategy(ABC):
    def __init__(
        self,
        molar_mass: Union[float, NDArray[np.float64]],
        diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-05,
        accommodation_coefficient: Union[float, NDArray[np.float64]] = 1.0,
        update_gases: bool = True,
    ): ...
```

### CondensationStrategy().first_order_mass_transport

[Show source in condensation_strategies.py:157](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/condensation/condensation_strategies.py#L157)

First-order mass transport coefficient per particle.

Calculate the first-order mass transport coefficient, K, for a given
particle based on the diffusion coefficient, radius, and vapor
transition correction factor.

#### Arguments

- `radius` - The radius of the particle [m].
- `temperature` - The temperature at which the first-order mass
transport coefficient is to be calculated.
- `pressure` - The pressure of the gas phase.
- `dynamic_viscosity` - The dynamic viscosity of the gas [Pa*s]. If not
provided, it will be calculated based on the temperature

#### Returns

The first-order mass transport coefficient per particle (m^3/s).

#### References

- Aerosol Modeling, Chapter 2, Equation 2.49 (excluding particle
    number)

#### Signature

```python
def first_order_mass_transport(
    self,
    radius: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    dynamic_viscosity: Optional[float] = None,
) -> Union[float, NDArray[np.float64]]: ...
```

### CondensationStrategy().knudsen_number

[Show source in condensation_strategies.py:122](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/condensation/condensation_strategies.py#L122)

The Knudsen number for a particle.

Calculate the Knudsen number based on the mean free path of the gas
molecules and the radius of the particle.

#### Arguments

- `radius` - The radius of the particle [m].
- `temperature` - The temperature of the gas [K].
- `pressure` - The pressure of the gas [Pa].
- `dynamic_viscosity` - The dynamic viscosity of the gas [Pa*s]. If
    not provided, it will be calculated based on the temperature

#### Returns

The Knudsen number, which is the ratio of the mean free path to
    the particle radius.

#### References

[Knudsen Number](https://en.wikipedia.org/wiki/Knudsen_number)

#### Signature

```python
def knudsen_number(
    self,
    radius: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    dynamic_viscosity: Optional[float] = None,
) -> Union[float, NDArray[np.float64]]: ...
```

### CondensationStrategy().mass_transfer_rate

[Show source in condensation_strategies.py:200](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/condensation/condensation_strategies.py#L200)

Mass transfer rate for a particle.

Calculate the mass transfer rate based on the difference in partial
pressure and the first-order mass transport coefficient.

#### Arguments

- `particle` - The particle for which the mass transfer rate is to be
    calculated.
- `gas_species` - The gas species with which the particle is in contact.
- `temperature` - The temperature at which the mass transfer rate
    is to be calculated.
- `pressure` - The pressure of the gas phase.
- `dynamic_viscosity` - The dynamic viscosity of the gas [Pa*s]. If not
    provided, it will be calculated based on the temperature

#### Returns

The mass transfer rate for the particle [kg/s].

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
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [GasSpecies](../../gas/species.md#gasspecies)
- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CondensationStrategy().mean_free_path

[Show source in condensation_strategies.py:91](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/condensation/condensation_strategies.py#L91)

Calculate the mean free path of the gas molecules based on the
temperature, pressure, and dynamic viscosity of the gas.

#### Arguments

- `temperature` - The temperature of the gas [K].
- `pressure` - The pressure of the gas [Pa].
- `dynamic_viscosity` - The dynamic viscosity of the gas [Pa*s]. If not
provided, it will be calculated based on the temperature

#### Returns

- `Union[float,` *NDArray[np.float64]]* - The mean free path of the gas
    molecules in meters (m).

#### References

Mean Free Path:
[Wikipedia](https://en.wikipedia.org/wiki/Mean_free_path)

#### Signature

```python
def mean_free_path(
    self, temperature: float, pressure: float, dynamic_viscosity: Optional[float] = None
) -> Union[float, NDArray[np.float64]]: ...
```

### CondensationStrategy().rate

[Show source in condensation_strategies.py:229](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/condensation/condensation_strategies.py#L229)

Calculate the rate of mass condensation for each particle due to
each condensable gas species.

The rate of condensation is determined based on the mass transfer rate,
which is a function of particle properties, gas species properties,
temperature, and pressure. This rate is then scaled by the
concentration of particles in the system to get the overall
condensation rate for each particle or bin.

#### Arguments

- `particle` *ParticleRepresentation* - Representation of the particles,
    including properties such as size, concentration, and mass.
- `gas_species` *GasSpecies* - The species of gas condensing onto the
    particles.
- `temperature` *float* - The temperature of the system in Kelvin.
- `pressure` *float* - The pressure of the system in Pascals.

#### Returns

An array of condensation rates for each particle, scaled by
    particle concentration.

#### Signature

```python
@abstractmethod
def rate(
    self,
    particle: ParticleRepresentation,
    gas_species: GasSpecies,
    temperature: float,
    pressure: float,
) -> NDArray[np.float64]: ...
```

#### See also

- [GasSpecies](../../gas/species.md#gasspecies)
- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CondensationStrategy().step

[Show source in condensation_strategies.py:261](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/condensation/condensation_strategies.py#L261)

Execute the condensation process for a given time step.

#### Arguments

- `particle` *ParticleRepresentation* - The particle to modify.
- `gas_species` *GasSpecies* - The gas species to condense onto the
    particle.
- `temperature` *float* - The temperature of the system in Kelvin.
- `pressure` *float* - The pressure of the system in Pascals.
- `time_step` *float* - The time step for the process in seconds.

#### Returns

- `(ParticleRepresentation,` *GasSpecies)* - The modified particle
    instance and the modified gas species instance.

#### Signature

```python
@abstractmethod
def step(
    self,
    particle: ParticleRepresentation,
    gas_species: GasSpecies,
    temperature: float,
    pressure: float,
    time_step: float,
) -> Tuple[ParticleRepresentation, GasSpecies]: ...
```

#### See also

- [GasSpecies](../../gas/species.md#gasspecies)
- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)
