# particula.gpu.nucleation_species_mapping

*module*

Explicit device configuration binding for native nucleation injection.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/nucleation_species_mapping.py#L1-L1)

## particula.gpu.nucleation_species_mapping.NucleationSpeciesBinding

*class* · *dataclass*

```python
class NucleationSpeciesBinding
```

Fixed explicit lane-map configuration for exact Warp particle/gas state.

Create with ``prepare_nucleation_species_map``. Particle and gas arrays
remain caller-owned. ``particle_to_gas`` and ``gas_to_particle`` are fresh
device int32 arrays containing -1 for unmapped lanes. Partitioning
is not baked into these maps: dispatch reads current full gas flags.
Replacing storage, metadata, names or mapping requires fresh preparation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/nucleation_species_mapping.py#L16-L57)

### particula.gpu.nucleation_species_mapping.NucleationSpeciesBinding.array_layouts

*attribute* · *instance attribute*

```python
array_layouts: tuple[tuple[Any, ...], ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/nucleation_species_mapping.py#L33-L33)

### particula.gpu.nucleation_species_mapping.NucleationSpeciesBinding.arrays

*attribute* · *instance attribute*

```python
arrays: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/nucleation_species_mapping.py#L32-L32)

### particula.gpu.nucleation_species_mapping.NucleationSpeciesBinding.device

*attribute* · *instance attribute*

```python
device: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/nucleation_species_mapping.py#L34-L34)

### particula.gpu.nucleation_species_mapping.NucleationSpeciesBinding.distribution_kind

*attribute* · *instance attribute*

```python
distribution_kind: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/nucleation_species_mapping.py#L31-L31)

### particula.gpu.nucleation_species_mapping.NucleationSpeciesBinding.gas

*attribute* · *instance attribute*

```python
gas: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/nucleation_species_mapping.py#L28-L28)

### particula.gpu.nucleation_species_mapping.NucleationSpeciesBinding.gas_names

*attribute* · *instance attribute*

```python
gas_names: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/nucleation_species_mapping.py#L30-L30)

### particula.gpu.nucleation_species_mapping.NucleationSpeciesBinding.gas_to_particle

*attribute* · *instance attribute*

```python
gas_to_particle: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/nucleation_species_mapping.py#L36-L36)

### particula.gpu.nucleation_species_mapping.NucleationSpeciesBinding.particle_to_gas

*attribute* · *instance attribute*

```python
particle_to_gas: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/nucleation_species_mapping.py#L35-L35)

### particula.gpu.nucleation_species_mapping.NucleationSpeciesBinding.particles

*attribute* · *instance attribute*

```python
particles: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/nucleation_species_mapping.py#L27-L27)

### particula.gpu.nucleation_species_mapping.NucleationSpeciesBinding.species_map

*attribute* · *instance attribute*

```python
species_map: NucleationSpeciesMap
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/nucleation_species_mapping.py#L29-L29)

### particula.gpu.nucleation_species_mapping.NucleationSpeciesBinding.__init__

*method*

```python
def __init__(particles: Any, gas: Any, species_map: NucleationSpeciesMap, gas_names: tuple[str, ...], distribution_kind: int, arrays: tuple[Any, ...], array_layouts: tuple[tuple[Any, ...], ...], device: Any, particle_to_gas: Any, gas_to_particle: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/nucleation_species_mapping.py#L0-L0)

### particula.gpu.nucleation_species_mapping.NucleationSpeciesBinding.validate

*method*

```python
def validate(particles: Any, gas: Any) -> None
```

Reject incompatible binding reuse before numerical preparation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/nucleation_species_mapping.py#L38-L57)

## particula.gpu.nucleation_species_mapping.prepare_nucleation_species_map

*function*

```python
def prepare_nucleation_species_map(particles: Any, gas: Any, *, gas_names: Sequence[str], species_map: NucleationSpeciesMap) -> NucleationSpeciesBinding
```

Explicitly upload validated lane configuration, never simulation state.

Preparation validates host provenance and storage metadata before the two
configuration uploads. It performs no physical validation, kernel launch,
synchronization, payload readback, or mutation of caller-owned arrays.
Direct scientific admission owns physical checks and algorithm support.

**Parameters**

- `particles` (`Any`) — Tagged resolved Warp particle carrier.
- `gas` (`Any`) — Complete Warp gas carrier, independently sized from particles.
- `gas_names` (`Sequence[str]`) — Actual full gas order retained by the caller on transfer.
- `species_map` (`NucleationSpeciesMap`) — Immutable process-owned expected order and lane pairs.

**Returns**

- (`NucleationSpeciesBinding`) — A fixed binding with newly owned device int32 forward/reverse maps.

**Raises**

- `TypeError` — If the process map has the wrong type.
- `ValueError` — If provenance, schema, indices or devices are incompatible.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/nucleation_species_mapping.py#L60-L128)
