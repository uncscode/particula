# Strategy

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Strategy

> Auto-generated documentation for [particula.next.dynamics.coagulation.strategy](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py) module.

## CoagulationStrategy

[Show source in strategy.py:26](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L26)

Abstract class for defining a coagulation strategy. This class defines the
methods that must be implemented by any coagulation strategy.

#### Methods

- `kernel` - Calculate the coagulation kernel.
- `loss_rate` - Calculate the coagulation loss rate.
- `gain_rate` - Calculate the coagulation gain rate.
- `net_rate` - Calculate the net coagulation rate.
- `diffusive_knudsen` - Calculate the diffusive Knudsen number.
- `coulomb_potential_ratio` - Calculate the Coulomb potential ratio.

#### Signature

```python
class CoagulationStrategy(ABC): ...
```

### CoagulationStrategy().coulomb_potential_ratio

[Show source in strategy.py:205](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L205)

Calculate the Coulomb potential ratio based on the particle properties
and temperature.

#### Arguments

- `particle` - The particles for which the Coulomb
    potential ratio is to be calculated.
- `temperature` - The temperature of the gas phase [K].

#### Returns

The Coulomb potential ratio for the particle
    [dimensionless].

#### Signature

```python
def coulomb_potential_ratio(
    self, particle: ParticleRepresentation, temperature: float
) -> NDArray[np.float64]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().diffusive_knudsen

[Show source in strategy.py:169](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L169)

Calculate the diffusive Knudsen number based on the particle
properties, temperature, and pressure.

#### Arguments

- `particle` - The particle for which the diffusive
    Knudsen number is to be calculated.
- `temperature` - The temperature of the gas phase [K].
- `pressure` - The pressure of the gas phase [Pa].

#### Returns

- `NDArray[np.float64]` - The diffusive Knudsen number for the particle
    [dimensionless].

#### Signature

```python
def diffusive_knudsen(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> NDArray[np.float64]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().dimensionless_kernel

[Show source in strategy.py:40](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L40)

Calculate the dimensionless coagulation kernel based on the particle
properties interactions,
diffusive Knudsen number and Coulomb potential

#### Arguments

- [CoagulationStrategy().diffusive_knudsen](#coagulationstrategydiffusive_knudsen) - The diffusive Knudsen number
    for the particle [dimensionless].
- [CoagulationStrategy().coulomb_potential_ratio](#coagulationstrategycoulomb_potential_ratio) - The Coulomb potential
    ratio for the particle [dimensionless].

#### Returns

The dimensionless coagulation kernel for the particle
    [dimensionless].

#### Signature

```python
@abstractmethod
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### CoagulationStrategy().friction_factor

[Show source in strategy.py:227](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L227)

Calculate the friction factor based on the particle properties,
temperature, and pressure.

#### Arguments

- `particle` - The particle for which the friction factor
    is to be calculated.
- `temperature` - The temperature of the gas phase [K].
- `pressure` - The pressure of the gas phase [Pa].

#### Returns

The friction factor for the particle [dimensionless].

#### Signature

```python
def friction_factor(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> NDArray[np.float64]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().gain_rate

[Show source in strategy.py:102](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L102)

Calculate the coagulation gain rate based on the particle radius,
distribution, and the coagulation kernel.

#### Arguments

- `particle` - The particle for which the coagulation
    gain rate is to be calculated.
- [CoagulationStrategy().kernel](#coagulationstrategykernel) - The coagulation kernel.

#### Returns

The coagulation gain rate for the particle [kg/s].

#### Notes

May be abstracted to a separate module when different coagulation
    strategies are implemented (super droplet).

#### Signature

```python
@abstractmethod
def gain_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().kernel

[Show source in strategy.py:62](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L62)

Calculate the coagulation kernel based on the particle properties,
temperature, and pressure.

#### Arguments

- `particle` - The particle for which the coagulation
    kernel is to be calculated.
- `temperature` - The temperature of the gas phase [K].
- `pressure` - The pressure of the gas phase [Pa].

#### Returns

The coagulation kernel for the particle [m^3/s].

#### Signature

```python
@abstractmethod
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().loss_rate

[Show source in strategy.py:83](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L83)

Calculate the coagulation loss rate based on the particle radius,
distribution, and the coagulation kernel.

#### Arguments

- `particle` - The particle for which the coagulation
    loss rate is to be calculated.
- [CoagulationStrategy().kernel](#coagulationstrategykernel) - The coagulation kernel.

#### Returns

The coagulation loss rate for the particle [kg/s].

#### Signature

```python
@abstractmethod
def loss_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().net_rate

[Show source in strategy.py:125](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L125)

Calculate the net coagulation rate based on the particle radius,
distribution, and the coagulation kernel.

#### Arguments

- `particle` - The particle class for which the
    coagulation net rate is to be calculated.
- `temperature` - The temperature of the gas phase [K].
- `pressure` - The pressure of the gas phase [Pa].

#### Returns

- `Union[float,` *NDArray[np.float64]]* - The net coagulation rate for the
    particle [kg/s].

#### Signature

```python
@abstractmethod
def net_rate(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### CoagulationStrategy().step

[Show source in strategy.py:147](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L147)

Perform a single step of the coagulation process.

#### Arguments

- `particle` - The particle for which the coagulation step
    is to be performed.
- `temperature` - The temperature of the gas phase [K].
- `pressure` - The pressure of the gas phase [Pa].
- `time_step` - The time step for the coagulation process [s].

#### Returns

- `ParticleRepresentation` - The particle after the coagulation step.

#### Signature

```python
@abstractmethod
def step(
    self,
    particle: ParticleRepresentation,
    temperature: float,
    pressure: float,
    time_step: float,
) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)



## ContinuousGeneralPDF

[Show source in strategy.py:488](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L488)

Continuous PDF coagulation strategy class. This class implements the
methods defined in the CoagulationStrategy abstract class. The kernel
strategy is passed as an argument to the class, should use a dimensionless
kernel representation.

#### Methods

- `kernel` - Calculate the coagulation kernel.
- `loss_rate` - Calculate the coagulation loss rate.
- `gain_rate` - Calculate the coagulation gain rate.
- `net_rate` - Calculate the net coagulation rate.

#### Signature

```python
class ContinuousGeneralPDF(CoagulationStrategy):
    def __init__(self, kernel_strategy: KernelStrategy): ...
```

#### See also

- [CoagulationStrategy](#coagulationstrategy)
- [KernelStrategy](./kernel.md#kernelstrategy)

### ContinuousGeneralPDF().dimensionless_kernel

[Show source in strategy.py:505](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L505)

#### Signature

```python
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### ContinuousGeneralPDF().gain_rate

[Show source in strategy.py:562](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L562)

#### Signature

```python
def gain_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ContinuousGeneralPDF().kernel

[Show source in strategy.py:515](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L515)

#### Signature

```python
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ContinuousGeneralPDF().loss_rate

[Show source in strategy.py:550](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L550)

#### Signature

```python
def loss_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ContinuousGeneralPDF().net_rate

[Show source in strategy.py:574](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L574)

#### Signature

```python
def net_rate(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ContinuousGeneralPDF().step

[Show source in strategy.py:592](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L592)

#### Signature

```python
def step(
    self,
    particle: ParticleRepresentation,
    temperature: float,
    pressure: float,
    time_step: float,
) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)



## DiscreteGeneral

[Show source in strategy.py:364](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L364)

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

[Show source in strategy.py:387](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L387)

#### Signature

```python
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### DiscreteGeneral().gain_rate

[Show source in strategy.py:443](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L443)

#### Signature

```python
def gain_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteGeneral().kernel

[Show source in strategy.py:397](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L397)

#### Signature

```python
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteGeneral().loss_rate

[Show source in strategy.py:432](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L432)

#### Signature

```python
def loss_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteGeneral().net_rate

[Show source in strategy.py:455](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L455)

#### Signature

```python
def net_rate(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteGeneral().step

[Show source in strategy.py:473](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L473)

#### Signature

```python
def step(
    self,
    particle: ParticleRepresentation,
    temperature: float,
    pressure: float,
    time_step: float,
) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)



## DiscreteSimple

[Show source in strategy.py:269](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L269)

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

[Show source in strategy.py:282](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L282)

#### Signature

```python
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### DiscreteSimple().gain_rate

[Show source in strategy.py:319](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L319)

#### Signature

```python
def gain_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteSimple().kernel

[Show source in strategy.py:294](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L294)

#### Signature

```python
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteSimple().loss_rate

[Show source in strategy.py:308](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L308)

#### Signature

```python
def loss_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteSimple().net_rate

[Show source in strategy.py:331](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L331)

#### Signature

```python
def net_rate(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### DiscreteSimple().step

[Show source in strategy.py:349](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L349)

#### Signature

```python
def step(
    self,
    particle: ParticleRepresentation,
    temperature: float,
    pressure: float,
    time_step: float,
) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)



## ParticleResolved

[Show source in strategy.py:607](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L607)

Particle-resolved coagulation strategy class. This class implements the
methods defined in the CoagulationStrategy abstract class. The kernel
strategy is passed as an argument to the class, should use a dimensionless
kernel representation.

#### Methods

- `kernel` - Calculate the coagulation kernel.
- `loss_rate` - Not implemented.
- `gain_rate` - Not implemented.
- `net_rate` - Not implemented.
- `step` - Perform a single step of the coagulation process.

#### Signature

```python
class ParticleResolved(CoagulationStrategy):
    def __init__(
        self,
        kernel_radius: Optional[NDArray[np.float64]] = None,
        kernel_bins_number: Optional[int] = None,
        kernel_bins_per_decade: int = 10,
    ): ...
```

#### See also

- [CoagulationStrategy](#coagulationstrategy)

### ParticleResolved().dimensionless_kernel

[Show source in strategy.py:672](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L672)

#### Signature

```python
def dimensionless_kernel(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### ParticleResolved().gain_rate

[Show source in strategy.py:712](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L712)

#### Signature

```python
def gain_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ParticleResolved().get_kernel_radius

[Show source in strategy.py:632](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L632)

Get the binning for the kernel radius.

If the kernel radius is not set, it will be calculated based on the
particle radius.

#### Arguments

- `particle` - The particle for which the kernel radius is to be
    calculated.

#### Returns

The kernel radius for the particle [m].

#### Signature

```python
def get_kernel_radius(self, particle: ParticleRepresentation) -> NDArray[np.float64]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ParticleResolved().kernel

[Show source in strategy.py:685](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L685)

#### Signature

```python
def kernel(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ParticleResolved().loss_rate

[Show source in strategy.py:702](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L702)

#### Signature

```python
def loss_rate(
    self, particle: ParticleRepresentation, kernel: NDArray[np.float64]
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ParticleResolved().net_rate

[Show source in strategy.py:722](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L722)

#### Signature

```python
def net_rate(
    self, particle: ParticleRepresentation, temperature: float, pressure: float
) -> Union[float, NDArray[np.float64]]: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)

### ParticleResolved().step

[Show source in strategy.py:733](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/strategy.py#L733)

#### Signature

```python
def step(
    self,
    particle: ParticleRepresentation,
    temperature: float,
    pressure: float,
    time_step: float,
) -> ParticleRepresentation: ...
```

#### See also

- [ParticleRepresentation](../../particles/representation.md#particlerepresentation)
