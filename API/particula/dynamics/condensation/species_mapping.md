# particula.dynamics.condensation.species_mapping

*module*

Explicit species provenance and read-only native condensation admission.

The map belongs to the process, not to particle or gas storage. Gas names
describe the complete gas ordering; pairs declare ordinary one-to-one mass
transfer. An omitted pair is explicitly nonparticipating. No chemical identity
is inferred from array widths or partitioning flags.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/species_mapping.py#L1-L1)

## particula.dynamics.condensation.species_mapping.CondensationSpeciesMap

*class* · *dataclass*

```python
class CondensationSpeciesMap
```

Immutable process-owned gas order and gas-to-particle lane pairs.

**Parameters**

- `gas_names` (`tuple[str, ...]`) — Complete expected gas order, including inert/unmapped gas.
- `gas_to_particle` (`tuple[tuple[int, int], ...]`) — Ordered ``(gas_index, particle_index)`` pairs. Indices must be nonnegative integers, with unique sources and targets. Empty pairs explicitly disable all transfer. Particle upper bounds are checked against current storage on every native admission.

**Raises**

- `ValueError` — If names or pairs are ambiguous or malformed.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/species_mapping.py#L20-L53)

### particula.dynamics.condensation.species_mapping.CondensationSpeciesMap.gas_names

*attribute* · *instance attribute*

```python
gas_names: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/species_mapping.py#L35-L35)

### particula.dynamics.condensation.species_mapping.CondensationSpeciesMap.gas_to_particle

*attribute* · *instance attribute*

```python
gas_to_particle: tuple[tuple[int, int], ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/species_mapping.py#L36-L36)

### particula.dynamics.condensation.species_mapping.CondensationSpeciesMap.__init__

*method*

```python
def __init__(gas_names: tuple[str, ...], gas_to_particle: tuple[tuple[int, int], ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/species_mapping.py#L0-L0)

### particula.dynamics.condensation.species_mapping.CondensationSpeciesMap.__post_init__

*method*

```python
def __post_init__() -> None
```

Detach caller sequences and reject ambiguous alignment.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/species_mapping.py#L38-L53)

## particula.dynamics.condensation.species_mapping._validate_gas_state

*function*

```python
def _validate_gas_state(gas: GasData, species_map: CondensationSpeciesMap) -> None
```

Validate full gas schema, physical state and ordered provenance.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/species_mapping.py#L82-L112)

## particula.dynamics.condensation.species_mapping._validate_pairs

*function*

```python
def _validate_pairs(values: Sequence[tuple[int, int]], width: int) -> list[tuple[int, int]]
```

Validate integer lane pairs and detach caller-owned sequences.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/species_mapping.py#L56-L79)

## particula.dynamics.condensation.species_mapping.condensation_population_weights

*function*

```python
def condensation_population_weights(particle: ParticleData) -> NDArray[np.float64]
```

Return fresh per-node inventory weights in m^-3 for one CPU box.

Resolved weights are counts/V; PMF weights are stored bin densities.
PDF weights multiply dn/dr by trapezoidal radius-node widths, so a sum of
weight times species mass equals the integral of the product. Neither the
radius grid nor raw concentration is modified.

**Parameters**

- `particle` (`ParticleData`) — Explicitly tagged particle data with SI storage.

**Returns**

- (`NDArray[np.float64]`) — One density weight per particle/bin/PDF node.

**Raises**

- `ValueError` — If representation validation fails or boxes differ from 1.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/species_mapping.py#L190-L222)

## particula.dynamics.condensation.species_mapping.validate_condensation_state

*function*

```python
def validate_condensation_state(particle: ParticleData, gas: GasData, species_map: CondensationSpeciesMap | None, temperature: float, pressure: float, skip_partitioning_indices: Sequence[int] | None = None) -> tuple[NDArray[np.intp], NDArray[np.intp]]
```

Validate current native state and return participating lane indices.

Particle representation validation supplies physical mass/count/density,
volume and PDF-grid checks. Gas storage is checked without calling its
coercing constructor. Returned arrays are fresh; all input storage and
metadata remain untouched, including on failure. Skip indices follow the
full gas order. Nonpartitioning or unmapped gas never transfers.

**Parameters**

- `particle` (`ParticleData`) — Explicitly tagged particle data, in SI units, with one box.
- `gas` (`GasData`) — Full ordered gas state; concentration in kg/m3, molar mass kg/mol.
- `species_map` (`CondensationSpeciesMap | None`) — Explicit expected gas order and transfer pairs.
- `temperature` (`float`) — Positive finite scalar temperature in K.
- `pressure` (`float`) — Positive finite scalar pressure in Pa.
- `skip_partitioning_indices` (`Sequence[int] | None`) (default: `None`) — Additional full-gas lanes to exclude.

**Returns**

- (`tuple[NDArray[np.intp], NDArray[np.intp]]`) — Participating gas and particle indices in declared pair order.

**Raises**

- `TypeError` — If containers or the process map have incorrect types.
- `ValueError` — If current state, configuration or physical inputs fail.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/species_mapping.py#L115-L187)
