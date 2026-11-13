# particula.dynamics.nucleation.species_mapping

*module*

Explicit full-gas order and injection alignment for native nucleation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/species_mapping.py#L1-L1)

## particula.dynamics.nucleation.species_mapping.NucleationSpeciesMap

*class* · *dataclass*

```python
class NucleationSpeciesMap
```

Own unique gas names and one-to-one gas-to-particle injection pairs.

Unmapped gas is preserved. Every positive injection mass must have a pair
and a partitioning-enabled gas lane. Particle-only lanes receive no source.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/species_mapping.py#L12-L53)

### particula.dynamics.nucleation.species_mapping.NucleationSpeciesMap.gas_names

*attribute* · *instance attribute*

```python
gas_names: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/species_mapping.py#L20-L20)

### particula.dynamics.nucleation.species_mapping.NucleationSpeciesMap.gas_to_particle

*attribute* · *instance attribute*

```python
gas_to_particle: tuple[tuple[int, int], ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/species_mapping.py#L21-L21)

### particula.dynamics.nucleation.species_mapping.NucleationSpeciesMap.__init__

*method*

```python
def __init__(gas_names: tuple[str, ...], gas_to_particle: tuple[tuple[int, int], ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/species_mapping.py#L0-L0)

### particula.dynamics.nucleation.species_mapping.NucleationSpeciesMap.__post_init__

*method*

```python
def __post_init__() -> None
```

Detach input sequences and reject ambiguous lane identities.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/species_mapping.py#L23-L53)

## particula.dynamics.nucleation.species_mapping.validate_nucleation_mapping

*function*

```python
def validate_nucleation_mapping(particles: ParticleData, gas: GasData, species_map: NucleationSpeciesMap | None, injection_mass: np.ndarray) -> np.ndarray
```

Validate alignment and return particle-lane source mass.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/species_mapping.py#L56-L95)
