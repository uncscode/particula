# Strategy

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Strategy

> Auto-generated documentation for [particula.next.dynamics.coagulation.strategy](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py) module.

## CoagulationStrategy

[Show source in strategy.py:20](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L20)

Abstract class for defining a coagulation strategy. This class defines the
methods that must be implemented by any coagulation strategy.

#### Methods

--------
- kernel (abstractmethod): Calculate the coagulation kernel.
- `-` *loss_rate* - Calculate the coagulation loss rate.
- `-` *gain_rate* - Calculate the coagulation gain rate.
- `-` *net_rate* - Calculate the net coagulation rate.
- `-` *diffusive_knudsen* - Calculate the diffusive Knudsen number.
- `-` *coulomb_potential_ratio* - Calculate the Coulomb potential ratio.

#### Signature

```python
class CoagulationStrategy(ABC): ...
```

### CoagulationStrategy().coulomb_potential_ratio

[Show source in strategy.py:200](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L200)

Calculate the Coulomb potential ratio based on the particle properties
and temperature.

#### Arguments

-----
- particle (Particle class): The particles for which the Coulomb
potential ratio is to be calculated.
- temperature (float): The temperature of the gas phase [K].

#### Returns

--------
- `-` *NDArray[np.float_]* - The Coulomb potential ratio for the particle
[dimensionless].

#### Signature

```python
def coulomb_potential_ratio(
    self, particle: ParticleRepresentation, temperature: float
) -> NDArray[np.float_]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().diffusive_knudsen

[Show source in strategy.py:159](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L159)

Calculate the diffusive Knudsen number based on the particle
properties, temperature, and pressure.

#### Arguments

-----
- particle (Particle class): The particle for which the diffusive
Knudsen number is to be calculated.
- temperature (float): The temperature of the gas phase [K].
- pressure (float): The pressure of the gas phase [Pa].

#### Returns

--------
- `-` *NDArray[np.float_]* - The diffusive Knudsen number for the particle
[dimensionless].

#### Signature

```python
def diffusive_knudsen(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> NDArray[np.float_]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().dimensionless_kernel

[Show source in strategy.py:35](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L35)

Calculate the dimensionless coagulation kernel based on the particle
properties interactions,
diffusive Knudsen number and Coulomb potential

#### Arguments

-----
- diffusive_knudsen (NDArray[np.float_]): The diffusive Knudsen number
for the particle [dimensionless].
- coulomb_potential_ratio (NDArray[np.float_]): The Coulomb potential
ratio for the particle [dimensionless].

#### Returns

--------
- `-` *NDArray[np.float_]* - The dimensionless coagulation kernel for the
particle [dimensionless].

#### Signature

```python
@abstractmethod
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float_],
    coulomb_potential_ratio: NDArray[np.float_],
) -> NDArray[np.float_]: ...
```

### CoagulationStrategy().friction_factor

[Show source in strategy.py:226](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L226)

Calculate the friction factor based on the particle properties,
temperature, and pressure.

#### Arguments

-----
- particle (Particle class): The particle for which the friction factor
is to be calculated.
- temperature (float): The temperature of the gas phase [K].
- pressure (float): The pressure of the gas phase [Pa].

#### Returns

--------
- `-` *NDArray[np.float_]* - The friction factor for the particle
[dimensionless].

#### Signature

```python
def friction_factor(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> NDArray[np.float_]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().gain_rate

[Show source in strategy.py:104](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L104)

Calculate the coagulation gain rate based on the particle radius,
distribution, and the coagulation kernel.

#### Arguments

-----
- particle (Particle class): The particle for which the coagulation
gain rate is to be calculated.
- kernel (NDArray[np.float_]): The coagulation kernel.

#### Returns

--------
- Union[float, NDArray[np.float_]]: The coagulation gain rate for the
particle [kg/s].

#### Notes

------
May be abstracted to a separate module when different coagulation
strategies are implemented (super droplet).

#### Signature

```python
@abstractmethod
def gain_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float_]
) -> Union[float, NDArray[np.float_]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().kernel

[Show source in strategy.py:59](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L59)

Calculate the coagulation kernel based on the particle properties,
temperature, and pressure.

#### Arguments

-----
- particle (Particle class): The particle for which the coagulation
kernel is to be calculated.
- temperature (float): The temperature of the gas phase [K].
- pressure (float): The pressure of the gas phase [Pa].

#### Returns

--------
- `-` *NDArray[np.float_]* - The coagulation kernel for the particle [m^3/s].

#### Signature

```python
@abstractmethod
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float_]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().loss_rate

[Show source in strategy.py:82](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L82)

Calculate the coagulation loss rate based on the particle radius,
distribution, and the coagulation kernel.

#### Arguments

-----
- particle (Particle class): The particle for which the coagulation
loss rate is to be calculated.
- kernel (NDArray[np.float_]): The coagulation kernel.

#### Returns

--------
- Union[float, NDArray[np.float_]]: The coagulation loss rate for the
particle [kg/s].

#### Signature

```python
@abstractmethod
def loss_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float_]
) -> Union[float, NDArray[np.float_]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().net_rate

[Show source in strategy.py:131](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L131)

Calculate the net coagulation rate based on the particle radius,
distribution, and the coagulation kernel.

#### Arguments

-----
- particle (Particle class): The particle class for which the
coagulation net rate is to be calculated.
- temperature (float): The temperature of the gas phase [K].
- pressure (float): The pressure of the gas phase [Pa].

#### Returns

--------
- Union[float, NDArray[np.float_]]: The net coagulation rate for the
particle [kg/s].

#### Signature

```python
def net_rate(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float_]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)



## ContinuousGeneralPDF

[Show source in strategy.py:426](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L426)

Continuous PDF coagulation strategy class. This class implements the
methods defined in the CoagulationStrategy abstract class. The kernel
strategy is passed as an argument to the class, should use a dimensionless
kernel representation.

#### Methods

--------
- `-` *kernel* - Calculate the coagulation kernel.
- `-` *loss_rate* - Calculate the coagulation loss rate.
- `-` *gain_rate* - Calculate the coagulation gain rate.
- `-` *net_rate* - Calculate the net coagulation rate.

#### Signature

```python
class ContinuousGeneralPDF(CoagulationStrategy):
    def __init__(self, kernel_strategy: KernelStrategy): ...
```

#### See also

- [CoagulationStrategy](#coagulationstrategy)
- [KernelStrategy](./kernel.md#kernelstrategy)

### ContinuousGeneralPDF().dimensionless_kernel

[Show source in strategy.py:444](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L444)

#### Signature

```python
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float_],
    coulomb_potential_ratio: NDArray[np.float_],
) -> NDArray[np.float_]: ...
```

### ContinuousGeneralPDF().gain_rate

[Show source in strategy.py:506](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L506)

#### Signature

```python
def gain_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float_]
) -> Union[float, NDArray[np.float_]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ContinuousGeneralPDF().kernel

[Show source in strategy.py:454](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L454)

#### Signature

```python
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float_]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ContinuousGeneralPDF().loss_rate

[Show source in strategy.py:494](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L494)

#### Signature

```python
def loss_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float_]
) -> Union[float, NDArray[np.float_]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)



## DiscreteGeneral

[Show source in strategy.py:330](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L330)

Discrete general coagulation strategy class. This class implements the
methods defined in the CoagulationStrategy abstract class. The kernel
strategy is passed as an argument to the class, to use a dimensionless
kernel representation.

#### Attributes

-----------
- `-` *kernel_strategy* - The kernel strategy to be used for the coagulation, from
the KernelStrategy class.

#### Methods

--------
- `-` *kernel* - Calculate the coagulation kernel.
- `-` *loss_rate* - Calculate the coagulation loss rate.
- `-` *gain_rate* - Calculate the coagulation gain rate.
- `-` *net_rate* - Calculate the net coagulation rate.

#### Signature

```python
class DiscreteGeneral(CoagulationStrategy):
    def __init__(self, kernel_strategy: KernelStrategy): ...
```

#### See also

- [CoagulationStrategy](#coagulationstrategy)
- [KernelStrategy](./kernel.md#kernelstrategy)

### DiscreteGeneral().dimensionless_kernel

[Show source in strategy.py:353](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L353)

#### Signature

```python
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float_],
    coulomb_potential_ratio: NDArray[np.float_],
) -> NDArray[np.float_]: ...
```

### DiscreteGeneral().gain_rate

[Show source in strategy.py:414](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L414)

#### Signature

```python
def gain_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float_]
) -> Union[float, NDArray[np.float_]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteGeneral().kernel

[Show source in strategy.py:363](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L363)

#### Signature

```python
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float_]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteGeneral().loss_rate

[Show source in strategy.py:403](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L403)

#### Signature

```python
def loss_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float_]
) -> Union[float, NDArray[np.float_]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)



## DiscreteSimple

[Show source in strategy.py:270](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L270)

Discrete Brownian coagulation strategy class. This class implements the
methods defined in the CoagulationStrategy abstract class.

#### Methods

--------
- `-` *kernel* - Calculate the coagulation kernel.
- `-` *loss_rate* - Calculate the coagulation loss rate.
- `-` *gain_rate* - Calculate the coagulation gain rate.
- `-` *net_rate* - Calculate the net coagulation rate.

#### Signature

```python
class DiscreteSimple(CoagulationStrategy): ...
```

#### See also

- [CoagulationStrategy](#coagulationstrategy)

### DiscreteSimple().dimensionless_kernel

[Show source in strategy.py:283](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L283)

#### Signature

```python
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float_],
    coulomb_potential_ratio: NDArray[np.float_],
) -> NDArray[np.float_]: ...
```

### DiscreteSimple().gain_rate

[Show source in strategy.py:318](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L318)

#### Signature

```python
def gain_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float_]
) -> Union[float, NDArray[np.float_]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteSimple().kernel

[Show source in strategy.py:293](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L293)

#### Signature

```python
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float_]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteSimple().loss_rate

[Show source in strategy.py:307](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L307)

#### Signature

```python
def loss_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float_]
) -> Union[float, NDArray[np.float_]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)
