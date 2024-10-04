# Aerosol

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Next](./index.md#next) / Aerosol

> Auto-generated documentation for [particula.next.aerosol](https://github.com/uncscode/particula/blob/main/particula/next/aerosol.py) module.

## Aerosol

[Show source in aerosol.py:14](https://github.com/uncscode/particula/blob/main/particula/next/aerosol.py#L14)

Collection of Gas and Particle objects.

A class for interacting with collections of Gas and Particle objects.
Allows for the representation and manipulation of an aerosol, which
is composed of various gases and particles.

#### Signature

```python
class Aerosol:
    def __init__(
        self,
        atmosphere: Atmosphere,
        particles: Union[ParticleRepresentation, List[ParticleRepresentation]],
    ): ...
```

#### See also

- [Atmosphere](gas/atmosphere.md#atmosphere)
- [ParticleRepresentation](particles/representation.md#particlerepresentation)

### Aerosol().__str__

[Show source in aerosol.py:39](https://github.com/uncscode/particula/blob/main/particula/next/aerosol.py#L39)

Returns a string representation of the aerosol.

#### Returns

- `str` - A string representation of the aerosol.

#### Signature

```python
def __str__(self) -> str: ...
```

### Aerosol().add_particle

[Show source in aerosol.py:75](https://github.com/uncscode/particula/blob/main/particula/next/aerosol.py#L75)

Adds a Particle instance to the aerosol.

#### Arguments

- `particle` - The Particle instance to add.

#### Signature

```python
def add_particle(self, particle: ParticleRepresentation): ...
```

#### See also

- [ParticleRepresentation](particles/representation.md#particlerepresentation)

### Aerosol().iterate_gas

[Show source in aerosol.py:51](https://github.com/uncscode/particula/blob/main/particula/next/aerosol.py#L51)

Returns an iterator for atmosphere species.

#### Returns

- `Iterator[GasSpecies]` - An iterator over the gas species type.

#### Signature

```python
def iterate_gas(self) -> Iterator[GasSpecies]: ...
```

#### See also

- [GasSpecies](gas/species.md#gasspecies)

### Aerosol().iterate_particle

[Show source in aerosol.py:59](https://github.com/uncscode/particula/blob/main/particula/next/aerosol.py#L59)

Returns an iterator for particle.

#### Returns

- `Iterator[Particle]` - An iterator over the particle type.

#### Signature

```python
def iterate_particle(self) -> Iterator[ParticleRepresentation]: ...
```

#### See also

- [ParticleRepresentation](particles/representation.md#particlerepresentation)

### Aerosol().replace_atmosphere

[Show source in aerosol.py:67](https://github.com/uncscode/particula/blob/main/particula/next/aerosol.py#L67)

Replaces the current Atmosphere instance with a new one.

#### Arguments

- `gas` - The instance to replace the current one.

#### Signature

```python
def replace_atmosphere(self, atmosphere: Atmosphere): ...
```

#### See also

- [Atmosphere](gas/atmosphere.md#atmosphere)
