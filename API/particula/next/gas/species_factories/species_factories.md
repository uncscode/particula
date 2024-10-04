# Species Factories

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Gas](./index.md#gas) / Species Factories

> Auto-generated documentation for [particula.next.gas.species_factories](https://github.com/uncscode/particula/blob/main/particula/next/gas/species_factories.py) module.

## GasSpeciesFactory

[Show source in species_factories.py:12](https://github.com/uncscode/particula/blob/main/particula/next/gas/species_factories.py#L12)

Factory class to create species builders

Factory class to create species builders for creating gas species.

#### Methods

- `get_builders` - Returns the mapping of strategy types to builder
instances.
- `get_strategy` - Gets the strategy instance
for the specified strategy type.
    - `strategy_type` - Type of species builder to use, can be
    'gas_species' or 'preset_gas_species'.
    - `parameters` - Parameters required for the
    builder, dependent on the chosen strategy type.

#### Returns

- `GasSpecies` - An instance of the specified GasSpecies.

#### Raises

- `ValueError` - If an unknown strategy type is provided.

#### Signature

```python
class GasSpeciesFactory(
    StrategyFactory[Union[GasSpeciesBuilder, PresetGasSpeciesBuilder], GasSpecies]
): ...
```

#### See also

- [GasSpeciesBuilder](./species_builders.md#gasspeciesbuilder)
- [GasSpecies](./species.md#gasspecies)
- [PresetGasSpeciesBuilder](./species_builders.md#presetgasspeciesbuilder)

### GasSpeciesFactory().get_builders

[Show source in species_factories.py:42](https://github.com/uncscode/particula/blob/main/particula/next/gas/species_factories.py#L42)

Returns the mapping of strategy types to builder instances.

#### Returns

A dictionary mapping strategy types to builder instances.
    - `-` *gas_species* - GasSpeciesBuilder
    - `-` *preset_gas_species* - PresetGasSpeciesBuilder

#### Signature

```python
def get_builders(self): ...
```
