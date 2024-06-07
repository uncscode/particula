# Aerosol

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Next](./index.md#next) / Aerosol

> Auto-generated documentation for [particula.next.aerosol](https://github.com/Gorkowski/particula/blob/main/particula/next/aerosol.py) module.

## Aerosol

[Show source in aerosol.py:13](https://github.com/Gorkowski/particula/blob/main/particula/next/aerosol.py#L13)

A class for interacting with collections of Gas and Particle objects.
Allows for the representation and manipulation of an aerosol, which
is composed of various gases and particles.

#### Signature

```python
class Aerosol:
    def __init__(
        self,
        gas: Atmosphere,
        particles: Union[ParticleRepresentation, List[ParticleRepresentation]],
    ): ...
```

#### See also

- [Atmosphere](gas/atmosphere.md#atmosphere)
- [ParticleRepresentation](particles/representation.md#particlerepresentation)

### Aerosol().add_gas

[Show source in aerosol.py:55](https://github.com/Gorkowski/particula/blob/main/particula/next/aerosol.py#L55)

Replaces the current Gas instance with a new one.

#### Arguments

- gas (Gas): The Gas instance to replace the current one.

#### Signature

```python
def add_gas(self, gas: Atmosphere): ...
```

#### See also

- [Atmosphere](gas/atmosphere.md#atmosphere)

### Aerosol().add_particle

[Show source in aerosol.py:64](https://github.com/Gorkowski/particula/blob/main/particula/next/aerosol.py#L64)

Adds a Particle instance to the aerosol.

#### Arguments

- particle (Particle): The Particle instance to add.

#### Signature

```python
def add_particle(self, particle: ParticleRepresentation): ...
```

#### See also

- [ParticleRepresentation](particles/representation.md#particlerepresentation)

### Aerosol().iterate_gas

[Show source in aerosol.py:37](https://github.com/Gorkowski/particula/blob/main/particula/next/aerosol.py#L37)

Returns an iterator for gas species.

#### Returns

- `Iterator[GasSpecies]` - An iterator over the gas species type.

#### Signature

```python
def iterate_gas(self) -> Iterator[GasSpecies]: ...
```

#### See also

- [GasSpecies](gas/species.md#gasspecies)

### Aerosol().iterate_particle

[Show source in aerosol.py:46](https://github.com/Gorkowski/particula/blob/main/particula/next/aerosol.py#L46)

Returns an iterator for particle.

#### Returns

- `Iterator[Particle]` - An iterator over the particle type.

#### Signature

```python
def iterate_particle(self) -> Iterator[ParticleRepresentation]: ...
```

#### See also

- [ParticleRepresentation](particles/representation.md#particlerepresentation)
