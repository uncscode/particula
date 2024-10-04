# Atmosphere Builders

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Gas](./index.md#gas) / Atmosphere Builders

> Auto-generated documentation for [particula.next.gas.atmosphere_builders](https://github.com/uncscode/particula/blob/main/particula/next/gas/atmosphere_builders.py) module.

## AtmosphereBuilder

[Show source in atmosphere_builders.py:17](https://github.com/uncscode/particula/blob/main/particula/next/gas/atmosphere_builders.py#L17)

Builder class for creating Atmosphere objects using a fluent interface.

This class provides methods to configure and build an Atmosphere object,
allowing for step-by-step setting of atmospheric properties and
species composition.

#### Attributes

- `temperature` - Temperature of the gas mixture in Kelvin.
- `total_pressure` *float* - Total pressure of the gas mixture in Pascals.
- `species` *list[GasSpecies]* - List of GasSpecies objects in the mixture.
    Starts empty.

#### Methods

- `set_temperature(temperature,temperature_units)` - Sets the temperature.
- `set_pressure(pressure,pressure_units)` - Sets the total pressure.
- `add_species(species)` - Adds a GasSpecies object to the gas mixture.
- `set_parameters(parameters)` - Sets multiple parameters from a dictionary.
- `build()` - Validates the set parameters and returns an Atmosphere object.

#### Signature

```python
class AtmosphereBuilder(BuilderABC, BuilderTemperatureMixin, BuilderPressureMixin):
    def __init__(self): ...
```

#### See also

- [BuilderABC](../abc_builder.md#builderabc)
- [BuilderPressureMixin](../builder_mixin.md#builderpressuremixin)
- [BuilderTemperatureMixin](../builder_mixin.md#buildertemperaturemixin)

### AtmosphereBuilder().add_species

[Show source in atmosphere_builders.py:49](https://github.com/uncscode/particula/blob/main/particula/next/gas/atmosphere_builders.py#L49)

Adds a GasSpecies object to the gas mixture.

#### Arguments

- `species` *GasSpecies* - The GasSpecies object to be added.

#### Returns

- [AtmosphereBuilder](#atmospherebuilder) - Instance of this builder for chaining.

#### Signature

```python
def add_species(self, species: GasSpecies) -> "AtmosphereBuilder": ...
```

#### See also

- [GasSpecies](./species.md#gasspecies)

### AtmosphereBuilder().build

[Show source in atmosphere_builders.py:61](https://github.com/uncscode/particula/blob/main/particula/next/gas/atmosphere_builders.py#L61)

Validates the configuration and constructs the Atmosphere object.

This method checks that all necessary conditions are met for a valid
Atmosphere instance(e.g., at least one species must be present) and
then initializes the Atmosphere.

#### Returns

- `Atmosphere` - The newly created Atmosphere object, configured as
specified.

#### Raises

- `ValueError` - If no species have been added to the mixture.

#### Signature

```python
def build(self) -> Atmosphere: ...
```

#### See also

- [Atmosphere](./atmosphere.md#atmosphere)
