# Atmosphere

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Gas](./index.md#gas) / Atmosphere

> Auto-generated documentation for [particula.next.gas.atmosphere](https://github.com/uncscode/particula/blob/main/particula/next/gas/atmosphere.py) module.

## Atmosphere

[Show source in atmosphere.py:8](https://github.com/uncscode/particula/blob/main/particula/next/gas/atmosphere.py#L8)

Represents a mixture of gas species under specific conditions.

This class represents the atmospheric environment by detailing properties
such as temperature and pressure, alongside a dynamic list of gas species
present.

#### Attributes

- `temperature` - Temperature of the gas mixture in Kelvin.
- `total_pressure` - Total atmospheric pressure of the mixture inPascals.
- `species` - List of GasSpecies objects representing the
    various species within the gas mixture.

#### Methods

- `add_species(self,` *species* - GasSpecies) -> None:
    Adds a GasSpecies object to the mixture.
- `remove_species(self,` *index* - int) -> None:
    Removes a GasSpecies object from the mixture based on its index.

#### Signature

```python
class Atmosphere: ...
```

### Atmosphere().__getitem__

[Show source in atmosphere.py:64](https://github.com/uncscode/particula/blob/main/particula/next/gas/atmosphere.py#L64)

Retrieves a gas species by index.

#### Arguments

- `index` - The index of the gas species to retrieve.

#### Returns

- `GasSpecies` - The gas species at the specified index.

#### Signature

```python
def __getitem__(self, index: int) -> GasSpecies: ...
```

#### See also

- [GasSpecies](./species.md#gasspecies)

### Atmosphere().__iter__

[Show source in atmosphere.py:55](https://github.com/uncscode/particula/blob/main/particula/next/gas/atmosphere.py#L55)

Allows iteration over the species in the gas mixture.

#### Returns

- `Iterator[GasSpecies]` - An iterator over the gas species objects.

#### Signature

```python
def __iter__(self): ...
```

### Atmosphere().__len__

[Show source in atmosphere.py:75](https://github.com/uncscode/particula/blob/main/particula/next/gas/atmosphere.py#L75)

Returns the number of species in the gas mixture.

#### Returns

- `int` - The number of gas species in the mixture.

#### Signature

```python
def __len__(self) -> int: ...
```

### Atmosphere().__str__

[Show source in atmosphere.py:83](https://github.com/uncscode/particula/blob/main/particula/next/gas/atmosphere.py#L83)

Provides a string representation of the Atmosphere object.

#### Returns

- `str` - A string that includes the temperature, pressure, and a
    list of species in the mixture.

#### Signature

```python
def __str__(self) -> str: ...
```

### Atmosphere().add_species

[Show source in atmosphere.py:32](https://github.com/uncscode/particula/blob/main/particula/next/gas/atmosphere.py#L32)

Adds a GasSpecies object to the mixture.

#### Arguments

- `gas_species` - The gas species to be added.

#### Signature

```python
def add_species(self, gas_species: GasSpecies) -> None: ...
```

#### See also

- [GasSpecies](./species.md#gasspecies)

### Atmosphere().remove_species

[Show source in atmosphere.py:40](https://github.com/uncscode/particula/blob/main/particula/next/gas/atmosphere.py#L40)

Removes a gas species from the mixture by its index.

#### Arguments

- `index` - Index of the gas species to remove. Must be within
            the current range of the list.

#### Raises

- `IndexError` - If the provided index is out of bounds.

#### Signature

```python
def remove_species(self, index: int) -> None: ...
```
