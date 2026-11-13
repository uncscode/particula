# particula.gpu.species_mapping

*module*

Explicit preparation of process-owned condensation lane maps on a device.

Gas names remain CPU-owned provenance. Callers supply their actual full order
at this boundary; it must match the process expectation. The returned binding
pins exact carriers and arrays and owns fresh device maps. These maps are
configuration, not physical state. They must remain unmodified while bound.
No name, lane, distribution, or device is inferred from equal array widths.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/species_mapping.py#L1-L1)

## particula.gpu.species_mapping._GAS_FIELDS

*attribute* · *module attribute*

```python
_GAS_FIELDS = ('concentration', 'molar_mass', 'partitioning', 'vapor_pressure')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/species_mapping.py#L26-L26)

## particula.gpu.species_mapping._PARTICLE_FIELDS

*attribute* · *module attribute*

```python
_PARTICLE_FIELDS = ('masses', 'concentration', 'charge', 'density', 'volume')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/species_mapping.py#L25-L25)

## particula.gpu.species_mapping.CondensationSpeciesBinding

*class* · *dataclass*

```python
class CondensationSpeciesBinding
```

Fixed explicit lane-map configuration for exact Warp particle/gas state.

Create with ``prepare_condensation_species_map``. Particle and gas arrays
remain caller-owned. ``particle_to_gas`` and ``gas_to_particle`` are fresh
device int32 arrays containing -1 for unmapped/skipped lanes. Partitioning
is not baked into these maps: dispatch reads current full gas flags.
Replacing storage, metadata, names or mapping requires fresh preparation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/species_mapping.py#L73-L115)

### particula.gpu.species_mapping.CondensationSpeciesBinding.array_layouts

*attribute* · *instance attribute*

```python
array_layouts: tuple[tuple[Any, ...], ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/species_mapping.py#L91-L91)

### particula.gpu.species_mapping.CondensationSpeciesBinding.arrays

*attribute* · *instance attribute*

```python
arrays: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/species_mapping.py#L90-L90)

### particula.gpu.species_mapping.CondensationSpeciesBinding.device

*attribute* · *instance attribute*

```python
device: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/species_mapping.py#L92-L92)

### particula.gpu.species_mapping.CondensationSpeciesBinding.distribution_kind

*attribute* · *instance attribute*

```python
distribution_kind: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/species_mapping.py#L89-L89)

### particula.gpu.species_mapping.CondensationSpeciesBinding.gas

*attribute* · *instance attribute*

```python
gas: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/species_mapping.py#L85-L85)

### particula.gpu.species_mapping.CondensationSpeciesBinding.gas_names

*attribute* · *instance attribute*

```python
gas_names: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/species_mapping.py#L87-L87)

### particula.gpu.species_mapping.CondensationSpeciesBinding.gas_to_particle

*attribute* · *instance attribute*

```python
gas_to_particle: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/species_mapping.py#L94-L94)

### particula.gpu.species_mapping.CondensationSpeciesBinding.particle_to_gas

*attribute* · *instance attribute*

```python
particle_to_gas: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/species_mapping.py#L93-L93)

### particula.gpu.species_mapping.CondensationSpeciesBinding.particles

*attribute* · *instance attribute*

```python
particles: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/species_mapping.py#L84-L84)

### particula.gpu.species_mapping.CondensationSpeciesBinding.skipped_gas_indices

*attribute* · *instance attribute*

```python
skipped_gas_indices: tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/species_mapping.py#L88-L88)

### particula.gpu.species_mapping.CondensationSpeciesBinding.species_map

*attribute* · *instance attribute*

```python
species_map: CondensationSpeciesMap
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/species_mapping.py#L86-L86)

### particula.gpu.species_mapping.CondensationSpeciesBinding.__init__

*method*

```python
def __init__(particles: Any, gas: Any, species_map: CondensationSpeciesMap, gas_names: tuple[str, ...], skipped_gas_indices: tuple[int, ...], distribution_kind: int, arrays: tuple[Any, ...], array_layouts: tuple[tuple[Any, ...], ...], device: Any, particle_to_gas: Any, gas_to_particle: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/species_mapping.py#L0-L0)

### particula.gpu.species_mapping.CondensationSpeciesBinding.validate

*method*

```python
def validate(particles: Any, gas: Any) -> None
```

Reject incompatible binding reuse before numerical preparation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/species_mapping.py#L96-L115)

## particula.gpu.species_mapping._binding_schema

*function*

```python
def _binding_schema(particles: Any, gas: Any) -> tuple[tuple[Any, ...], Any]
```

Check complete carrier schemas and a common device without readback.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/species_mapping.py#L34-L70)

## particula.gpu.species_mapping._layout

*function*

```python
def _layout(array: Any) -> tuple[Any, ...]
```

Freeze metadata that could otherwise change behind an array identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/species_mapping.py#L29-L31)

## particula.gpu.species_mapping.prepare_condensation_species_map

*function*

```python
def prepare_condensation_species_map(particles: Any, gas: Any, *, gas_names: Sequence[str], species_map: CondensationSpeciesMap, skip_partitioning_indices: Sequence[int] = ()) -> CondensationSpeciesBinding
```

Explicitly upload validated lane configuration, never simulation state.

Preparation validates host provenance and storage metadata before the two
configuration uploads. It performs no physical validation, kernel launch,
synchronization, payload readback, or mutation of caller-owned arrays.
Direct scientific admission owns physical checks and algorithm support.

**Parameters**

- `particles` (`Any`) — Tagged discrete or resolved Warp particle carrier.
- `gas` (`Any`) — Complete Warp gas carrier, independently sized from particles.
- `gas_names` (`Sequence[str]`) — Actual full gas order retained by the caller on transfer.
- `species_map` (`CondensationSpeciesMap`) — Immutable process-owned expected order and lane pairs.
- `skip_partitioning_indices` (`Sequence[int]`) (default: `()`) — Optional full gas indices to exclude.

**Returns**

- (`CondensationSpeciesBinding`) — A fixed binding with newly owned device int32 forward/reverse maps.

**Raises**

- `TypeError` — If the process map has the wrong type.
- `ValueError` — If provenance, schema, indices or devices are incompatible.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/species_mapping.py#L118-L201)
