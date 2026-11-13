# particula.dynamics.nucleation.nucleation_configuration

*module*

Define immutable public construction nucleation source-selection metadata.

This module only selects an already-built potential-rate strategy and precursor
species index, with an explicit full-gas map for native execution. It does not
import or provide concrete demand planning/source transaction
``particle_source`` finalization, transaction, or state-mutation helpers.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_configuration.py#L1-L1)

## particula.dynamics.nucleation.nucleation_configuration.NucleationSourceConfig

*class* · *dataclass*

```python
class NucleationSourceConfig
```

Select a public construction nucleation strategy and its precursor
    species index.

This record is metadata only; source-demand finalization and mutation remain
at concrete demand planning/source transaction boundaries.

**Attributes**

- `strategy` (`NucleationStrategy`) — Immutable activation or kinetic potential-rate strategy.
- `precursor_index` (`int`) — Nonnegative index of the precursor species selected for a later concrete source operation.
- `species_map` (`NucleationSpeciesMap | None`) — Explicit native gas order and injection mapping; None is retained only for transitional legacy callers.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_configuration.py#L21-L65)

### particula.dynamics.nucleation.nucleation_configuration.NucleationSourceConfig.precursor_index

*attribute* · *instance attribute*

```python
precursor_index: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_configuration.py#L38-L38)

### particula.dynamics.nucleation.nucleation_configuration.NucleationSourceConfig.species_map

*attribute* · *class attribute* · *instance attribute*

```python
species_map: NucleationSpeciesMap | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_configuration.py#L39-L39)

### particula.dynamics.nucleation.nucleation_configuration.NucleationSourceConfig.strategy

*attribute* · *instance attribute*

```python
strategy: NucleationStrategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_configuration.py#L37-L37)

### particula.dynamics.nucleation.nucleation_configuration.NucleationSourceConfig.__init__

*method*

```python
def __init__(strategy: NucleationStrategy, precursor_index: int, species_map: NucleationSpeciesMap | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_configuration.py#L0-L0)

### particula.dynamics.nucleation.nucleation_configuration.NucleationSourceConfig.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the supported strategy types and nonnegative index.

**Raises**

- `TypeError` — If ``precursor_index`` is not a non-Boolean integer.
- `ValueError` — If the strategy is unsupported or the index is negative.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_configuration.py#L41-L65)
