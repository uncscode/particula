# particula.dynamics.nucleation.particle_source

*module*

Plan and atomically commit CPU particle-source transactions.

The concrete-module-only demand planning boundary converts survival-adjusted
potential event
rates [#/m³/s] and a duration [s] into immutable, gas-admitted source-demand
records without accessing particle capacity or mutating caller state. The source
transaction
boundary consumes those records, stages slot activation and exhaustion policy on
a private ``ParticleData`` copy, and writes validated particle and gas arrays
atomically. Source transaction scales pre-existing particle and gas
concentrations for
selected representative-volume rows before removing finalized source mass.
Neither demand planning nor source transaction is exported through public
package namespaces.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L1-L1)

## particula.dynamics.nucleation.particle_source.FinalizedSourceDiagnostics

*class* · *dataclass*

```python
class FinalizedSourceDiagnostics
```

Own immutable diagnostics from a committed source transaction source
    transaction.

All arrays are fresh read-only copies. ``event_count_units`` is ``count``
for native resolved storage and ``m^-3`` for legacy. Gas mass fields are
[kg/m³]; slot counts and policy codes are ``int32``. Native conservation
residuals compare mapped particle-plus-gas density before and after, with
particle-only lanes included. Legacy residuals use the scaled-domain sum.

**Attributes**

- `potential_event_count` (`NDArray[np.float64]`) — demand planning potential event count, shape ``(n_boxes,)``.
- `gas_admitted_event_count` (`NDArray[np.float64]`) — demand planning gas-admitted event count, shape ``(n_boxes,)``.
- `represented_event_count` (`NDArray[np.float64]`) — Final event count represented in particle storage, shape ``(n_boxes,)``.
- `gas_limited_event_count` (`NDArray[np.float64]`) — demand planning event count excluded by gas inventory, shape ``(n_boxes,)``.
- `representation_reduction_event_count` (`NDArray[np.float64]`) — Event count removed by representative-volume scaling, shape ``(n_boxes,)``.
- `residual_event_count` (`NDArray[np.float64]`) — Unpackaged final source demand, shape ``(n_boxes,)``; successful transactions report zero.
- `limiting_species_index` (`NDArray[np.int32]`) — demand planning limiting gas-species index or ``-1``, shape ``(n_boxes,)``.
- `gas_mass_removed` (`NDArray[np.float64]`) — Final source mass removed from gas, shape ``(n_boxes, n_species)``.
- `requested_slot_count` (`NDArray[np.int32]`) — Final equal-weight source-slot request count, shape ``(n_boxes,)``.
- `activated_slot_count` (`NDArray[np.int32]`) — Number of source slots activated, shape ``(n_boxes,)``.
- `released_slot_count` (`NDArray[np.int32]`) — Number of slots released by resampling, shape ``(n_boxes,)``.
- `exhaustion_policy_code` (`NDArray[np.int32]`) — Resolved exhaustion-policy code, shape ``(n_boxes,)``.
- `representative_volume_scale` (`NDArray[np.float64]`) — Applied representative-volume scale, shape ``(n_boxes,)``.
- `conservation_residual` (`NDArray[np.float64]`) — Final particle-plus-gas scaled-domain mass residual, shape ``(n_boxes, n_species)``.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L511-L605)

### particula.dynamics.nucleation.particle_source.FinalizedSourceDiagnostics.activated_slot_count

*attribute* · *instance attribute*

```python
activated_slot_count: NDArray[np.int32]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L565-L565)

### particula.dynamics.nucleation.particle_source.FinalizedSourceDiagnostics.conservation_residual

*attribute* · *instance attribute*

```python
conservation_residual: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L569-L569)

### particula.dynamics.nucleation.particle_source.FinalizedSourceDiagnostics.event_count_units

*attribute* · *class attribute* · *instance attribute*

```python
event_count_units: str = 'm^-3'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L570-L570)

### particula.dynamics.nucleation.particle_source.FinalizedSourceDiagnostics.exhaustion_policy_code

*attribute* · *instance attribute*

```python
exhaustion_policy_code: NDArray[np.int32]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L567-L567)

### particula.dynamics.nucleation.particle_source.FinalizedSourceDiagnostics.gas_admitted_event_count

*attribute* · *instance attribute*

```python
gas_admitted_event_count: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L557-L557)

### particula.dynamics.nucleation.particle_source.FinalizedSourceDiagnostics.gas_limited_event_count

*attribute* · *instance attribute*

```python
gas_limited_event_count: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L559-L559)

### particula.dynamics.nucleation.particle_source.FinalizedSourceDiagnostics.gas_mass_removed

*attribute* · *instance attribute*

```python
gas_mass_removed: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L563-L563)

### particula.dynamics.nucleation.particle_source.FinalizedSourceDiagnostics.limiting_species_index

*attribute* · *instance attribute*

```python
limiting_species_index: NDArray[np.int32]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L562-L562)

### particula.dynamics.nucleation.particle_source.FinalizedSourceDiagnostics.potential_event_count

*attribute* · *instance attribute*

```python
potential_event_count: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L556-L556)

### particula.dynamics.nucleation.particle_source.FinalizedSourceDiagnostics.released_slot_count

*attribute* · *instance attribute*

```python
released_slot_count: NDArray[np.int32]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L566-L566)

### particula.dynamics.nucleation.particle_source.FinalizedSourceDiagnostics.representation_reduction_event_count

*attribute* · *instance attribute*

```python
representation_reduction_event_count: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L560-L560)

### particula.dynamics.nucleation.particle_source.FinalizedSourceDiagnostics.representative_volume_scale

*attribute* · *instance attribute*

```python
representative_volume_scale: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L568-L568)

### particula.dynamics.nucleation.particle_source.FinalizedSourceDiagnostics.represented_event_count

*attribute* · *instance attribute*

```python
represented_event_count: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L558-L558)

### particula.dynamics.nucleation.particle_source.FinalizedSourceDiagnostics.requested_slot_count

*attribute* · *instance attribute*

```python
requested_slot_count: NDArray[np.int32]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L564-L564)

### particula.dynamics.nucleation.particle_source.FinalizedSourceDiagnostics.residual_event_count

*attribute* · *instance attribute*

```python
residual_event_count: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L561-L561)

### particula.dynamics.nucleation.particle_source.FinalizedSourceDiagnostics.__init__

*method*

```python
def __init__(potential_event_count: NDArray[np.float64], gas_admitted_event_count: NDArray[np.float64], represented_event_count: NDArray[np.float64], gas_limited_event_count: NDArray[np.float64], representation_reduction_event_count: NDArray[np.float64], residual_event_count: NDArray[np.float64], limiting_species_index: NDArray[np.int32], gas_mass_removed: NDArray[np.float64], requested_slot_count: NDArray[np.int32], activated_slot_count: NDArray[np.int32], released_slot_count: NDArray[np.int32], exhaustion_policy_code: NDArray[np.int32], representative_volume_scale: NDArray[np.float64], conservation_residual: NDArray[np.float64], event_count_units: str = 'm^-3') -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L0-L0)

### particula.dynamics.nucleation.particle_source.FinalizedSourceDiagnostics.__post_init__

*method*

```python
def __post_init__() -> None
```

Defensively own all finalized diagnostic arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L572-L605)

## particula.dynamics.nucleation.particle_source.ParticleSourceCommitConfig

*class* · *dataclass*

```python
class ParticleSourceCommitConfig
```

Own controls and representative-volume inputs for a source transaction
    transaction.

The configuration owns read-only float64 public construction sidecars. Their
length is
checked against the demand planning box count by ``commit_particle_source``.
The four
resampling bounds intentionally require exact Python ``float`` values to
match the exhaustion-policy boundary.

**Attributes**

- `maximum_slot_weight` (`float`) — Positive maximum represented events per activated slot: counts for native resolved storage, #/m³ for legacy.
- `source_charge` (`float`) — Finite charge assigned to each activated source slot [elementary-charge counts].
- `exhaustion_controls` (`ExhaustionControls`) — Immutable controls selecting exhaustion policies.
- `requested_scale` (`NDArray[np.float64]`) — Requested representative-volume scale per box, shape ``(n_boxes,)``. This record owns a read-only float64 copy.
- `minimum_scale` (`NDArray[np.float64]`) — Minimum permitted representative-volume scale per box, shape ``(n_boxes,)``. This record owns a read-only float64 copy.
- `minimum_volume` (`NDArray[np.float64]`) — Minimum particle representative volume per box [m³], shape ``(n_boxes,)``. This record owns a read-only float64 copy.
- `radius_cubed_relative_error` (`float`) — Permitted resampling relative error for radius cubed.
- `mean_radius_relative_error` (`float`) — Permitted resampling relative error for mean radius.
- `surface_relative_error` (`float`) — Permitted resampling relative error for surface area.
- `diversity_absolute_error` (`float`) — Permitted resampling absolute diversity error.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L429-L508)

### particula.dynamics.nucleation.particle_source.ParticleSourceCommitConfig.diversity_absolute_error

*attribute* · *class attribute* · *instance attribute*

```python
diversity_absolute_error: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L471-L471)

### particula.dynamics.nucleation.particle_source.ParticleSourceCommitConfig.exhaustion_controls

*attribute* · *instance attribute*

```python
exhaustion_controls: ExhaustionControls
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L464-L464)

### particula.dynamics.nucleation.particle_source.ParticleSourceCommitConfig.maximum_slot_weight

*attribute* · *instance attribute*

```python
maximum_slot_weight: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L462-L462)

### particula.dynamics.nucleation.particle_source.ParticleSourceCommitConfig.mean_radius_relative_error

*attribute* · *class attribute* · *instance attribute*

```python
mean_radius_relative_error: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L469-L469)

### particula.dynamics.nucleation.particle_source.ParticleSourceCommitConfig.minimum_scale

*attribute* · *instance attribute*

```python
minimum_scale: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L466-L466)

### particula.dynamics.nucleation.particle_source.ParticleSourceCommitConfig.minimum_volume

*attribute* · *instance attribute*

```python
minimum_volume: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L467-L467)

### particula.dynamics.nucleation.particle_source.ParticleSourceCommitConfig.radius_cubed_relative_error

*attribute* · *class attribute* · *instance attribute*

```python
radius_cubed_relative_error: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L468-L468)

### particula.dynamics.nucleation.particle_source.ParticleSourceCommitConfig.requested_scale

*attribute* · *instance attribute*

```python
requested_scale: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L465-L465)

### particula.dynamics.nucleation.particle_source.ParticleSourceCommitConfig.source_charge

*attribute* · *instance attribute*

```python
source_charge: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L463-L463)

### particula.dynamics.nucleation.particle_source.ParticleSourceCommitConfig.species_map

*attribute* · *class attribute* · *instance attribute*

```python
species_map: NucleationSpeciesMap | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L472-L472)

### particula.dynamics.nucleation.particle_source.ParticleSourceCommitConfig.surface_relative_error

*attribute* · *class attribute* · *instance attribute*

```python
surface_relative_error: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L470-L470)

### particula.dynamics.nucleation.particle_source.ParticleSourceCommitConfig.__init__

*method*

```python
def __init__(maximum_slot_weight: float, source_charge: float, exhaustion_controls: ExhaustionControls, requested_scale: NDArray[np.float64], minimum_scale: NDArray[np.float64], minimum_volume: NDArray[np.float64], radius_cubed_relative_error: float = 1.0, mean_radius_relative_error: float = 1.0, surface_relative_error: float = 1.0, diversity_absolute_error: float = 1.0, species_map: NucleationSpeciesMap | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L0-L0)

### particula.dynamics.nucleation.particle_source.ParticleSourceCommitConfig.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate scalar controls and defensively own public construction
inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L474-L508)

## particula.dynamics.nucleation.particle_source.PotentialEventData

*class* · *dataclass*

```python
class PotentialEventData
```

Potential particle-formation rate and source duration.

**Attributes**

- `potential_rate` (`NDArray[np.float64]`) — Survival-adjusted event rate [#/m³/s] with shape ``(n_boxes,)``. The record owns a fresh read-only float64 copy.
- `duration` (`float`) — Finite nonnegative source duration [s].

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L121-L160)

### particula.dynamics.nucleation.particle_source.PotentialEventData.duration

*attribute* · *instance attribute*

```python
duration: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L132-L132)

### particula.dynamics.nucleation.particle_source.PotentialEventData.potential_rate

*attribute* · *instance attribute*

```python
potential_rate: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L131-L131)

### particula.dynamics.nucleation.particle_source.PotentialEventData.__init__

*method*

```python
def __init__(potential_rate: NDArray[np.float64], duration: float) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L0-L0)

### particula.dynamics.nucleation.particle_source.PotentialEventData.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate and defensively own the potential-source inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L134-L160)

## particula.dynamics.nucleation.particle_source.SourceDemandData

*class* · *dataclass*

```python
class SourceDemandData
```

Immutable demand planning gas mass demand for one common event count per
    box.

**Attributes**

- `per_event_mass` (`NDArray[np.float64]`) — Species mass per formed event [kg/event] with shape ``(n_species,)``. The record owns a fresh read-only float64 copy.
- `gas_mass_removed` (`NDArray[np.float64]`) — Provisional gas demand [kg/m³], shape ``(n_boxes, n_species)``. The record owns a fresh read-only float64 copy. Demand planning does not apply this demand to gas.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L163-L195)

### particula.dynamics.nucleation.particle_source.SourceDemandData.gas_mass_removed

*attribute* · *instance attribute*

```python
gas_mass_removed: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L177-L177)

### particula.dynamics.nucleation.particle_source.SourceDemandData.per_event_mass

*attribute* · *instance attribute*

```python
per_event_mass: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L176-L176)

### particula.dynamics.nucleation.particle_source.SourceDemandData.__init__

*method*

```python
def __init__(per_event_mass: NDArray[np.float64], gas_mass_removed: NDArray[np.float64]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L0-L0)

### particula.dynamics.nucleation.particle_source.SourceDemandData.__post_init__

*method*

```python
def __post_init__() -> None
```

Defensively own immutable float64 output payloads.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L179-L195)

## particula.dynamics.nucleation.particle_source.SourceDiagnostics

*class* · *dataclass*

```python
class SourceDiagnostics
```

Immutable source counts and limiting-species diagnostics.

All payloads are fresh, read-only arrays owned by this record. A
limiting-species index of ``-1`` means no limiter is reported, including
zero-potential and zero-admitted rows.

**Attributes**

- `potential_event_count` (`NDArray[np.float64]`) — Potential events [#/m³] with shape ``(n_boxes,)``.
- `gas_admitted_event_count` (`NDArray[np.float64]`) — Inventory-admitted events [#/m³] with shape ``(n_boxes,)``.
- `gas_limited_event_count` (`NDArray[np.float64]`) — Potential events not admitted by gas [#/m³] with shape ``(n_boxes,)``.
- `limiting_species_index` (`NDArray[np.int32]`) — Participating gas species that limits each box, with shape ``(n_boxes,)`` and ``-1`` as the no-limiting sentinel.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L198-L234)

### particula.dynamics.nucleation.particle_source.SourceDiagnostics.gas_admitted_event_count

*attribute* · *instance attribute*

```python
gas_admitted_event_count: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L218-L218)

### particula.dynamics.nucleation.particle_source.SourceDiagnostics.gas_limited_event_count

*attribute* · *instance attribute*

```python
gas_limited_event_count: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L219-L219)

### particula.dynamics.nucleation.particle_source.SourceDiagnostics.limiting_species_index

*attribute* · *instance attribute*

```python
limiting_species_index: NDArray[np.int32]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L220-L220)

### particula.dynamics.nucleation.particle_source.SourceDiagnostics.potential_event_count

*attribute* · *instance attribute*

```python
potential_event_count: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L217-L217)

### particula.dynamics.nucleation.particle_source.SourceDiagnostics.__init__

*method*

```python
def __init__(potential_event_count: NDArray[np.float64], gas_admitted_event_count: NDArray[np.float64], gas_limited_event_count: NDArray[np.float64], limiting_species_index: NDArray[np.int32]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L0-L0)

### particula.dynamics.nucleation.particle_source.SourceDiagnostics.__post_init__

*method*

```python
def __post_init__() -> None
```

Defensively own immutable output payloads.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L222-L234)

## particula.dynamics.nucleation.particle_source._is_real_scalar

*function*

```python
def _is_real_scalar(value: object) -> bool
```

Return whether a value is a non-Boolean real scalar.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L112-L118)

## particula.dynamics.nucleation.particle_source._readonly_copy

*function*

```python
def _readonly_copy(values: object, dtype: type[np.generic], name: str, ndim: int | None = None) -> NDArray[np.generic]
```

Copy an array payload into owned, read-only storage.

This helper ensures that frozen record attributes cannot be changed through
an aliased caller-owned NumPy array.

**Parameters**

- `values` (`object`) — Array-compatible payload.
- `dtype` (`type[np.generic]`) — Required output NumPy dtype.
- `name` (`str`) — Field name for validation errors.
- `ndim` (`int | None`) (default: `None`) — Optional required number of dimensions.

**Returns**

- (`NDArray[np.generic]`) — A fresh, owned, read-only NumPy array with the requested dtype.

**Raises**

- `TypeError` — If the value cannot be converted to the requested dtype.
- `ValueError` — If the array rank differs from ``ndim``.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L50-L87)

## particula.dynamics.nucleation.particle_source._readonly_vector

*function*

```python
def _readonly_vector(values: object, dtype: type[np.generic], name: str) -> NDArray[np.generic]
```

Copy a rank-one record payload into owned, read-only storage.

**Parameters**

- `values` (`object`) — Array-compatible rank-one payload.
- `dtype` (`type[np.generic]`) — Required output NumPy dtype.
- `name` (`str`) — Field name for validation errors.

**Returns**

- (`NDArray[np.generic]`) — A fresh, owned, read-only rank-one NumPy array.

**Raises**

- `TypeError` — If ``values`` cannot be converted to ``dtype``.
- `ValueError` — If ``values`` is not rank one.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L90-L109)

## particula.dynamics.nucleation.particle_source._request_counts

*function*

```python
def _request_counts(event_count: NDArray[np.float64], maximum_slot_weight: float, capacity: int, *, enforce_capacity: bool = True) -> NDArray[np.int32]
```

Calculate checked equal-weight source-slot requests per box.

A positive event count requests ``ceil(event_count / maximum_slot_weight)``
slots; zero count requests zero slots.

**Parameters**

- `event_count` (`NDArray[np.float64]`) — Final or provisional represented event count [#/m³].
- `maximum_slot_weight` (`float`) — Positive maximum represented events per slot [#/m³].
- `capacity` (`int`) — Fixed particle-slot capacity available in each box.
- `enforce_capacity` (`bool`) (default: `True`) — Whether to reject requests above ``capacity``. Policy resolution uses unchecked provisional requests before scaling; activation always enforces capacity.

**Returns**

- (`NDArray[np.int32]`) — Int32 source-slot request count for each box.

**Raises**

- `ValueError` — If the derived request count is nonfinite, negative, or
exceeds fixed particle capacity when ``enforce_capacity`` is true.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L858-L900)

## particula.dynamics.nucleation.particle_source._validate_commit_inputs

*function*

```python
def _validate_commit_inputs(demand: object, diagnostics: object, particles: object, gas: object, config: object) -> tuple[SourceDemandData, SourceDiagnostics, ParticleData, GasData, ParticleSourceCommitConfig]
```

Validate immutable demand planning records and mutable source transaction
    state before staging.

This read-only preflight verifies record consistency, writable particle and
gas schemas, nonaliasing, and representative-volume sidecars. It performs
no caller-visible mutation, preserving the transaction's all-or-nothing
boundary if validation fails.

**Parameters**

- `demand` (`object`) — Candidate demand planning per-event mass and provisional gas-demand record.
- `diagnostics` (`object`) — Candidate demand planning event-count diagnostic record.
- `particles` (`object`) — Candidate writable fixed-capacity particle container.
- `gas` (`object`) — Candidate writable gas inventory.
- `config` (`object`) — Candidate source transaction capacity and scaling configuration.

**Returns**

- (`SourceDemandData`) — Validated, type-narrowed demand planning records, particle data, gas
- (`SourceDiagnostics`) — data, and
- (`ParticleData`) — transaction configuration.

**Raises**

- `TypeError` — If a top-level boundary object has an invalid type.
- `ValueError` — If records, schemas, aliasing, physical values, or scaling
controls violate the source transaction transaction boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L679-L855)

## particula.dynamics.nucleation.particle_source._validate_commit_particle_schema

*function*

```python
def _validate_commit_particle_schema(particles: object) -> ParticleData
```

Validate writable, physically valid, non-overlapping source transaction
    particle storage.

**Parameters**

- `particles` (`object`) — Candidate fixed-capacity particle container.

**Returns**

- (`ParticleData`) — The validated particle container without modifying its arrays.

**Raises**

- `TypeError` — If ``particles`` or a required field has an invalid type.
- `ValueError` — If field schemas, mutability, aliasing, or physical values
violate the source transaction transaction boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L608-L676)

## particula.dynamics.nucleation.particle_source._validate_gas

*function*

```python
def _validate_gas(gas: GasData, n_boxes: int, n_species: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Validate gas schema and physical inventory inputs without mutation.

**Parameters**

- `gas` (`GasData`) — Gas inventory whose molar masses are [kg/mol] and concentrations are [kg/m³].
- `n_boxes` (`int`) — Required number of gas concentration rows.
- `n_species` (`int`) — Required number of gas species columns.

**Returns**

- (`NDArray[np.float64]`) — Float64 views or conversions of validated molar masses [kg/mol] and
- (`NDArray[np.float64]`) — concentrations [kg/m³]. Neither result is written by this module.

**Raises**

- `ValueError` — If gas schemas, dtypes, or physical values are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L237-L291)

## particula.dynamics.nucleation.particle_source._weighted_particle_mass

*function*

```python
def _weighted_particle_mass(particles: ParticleData) -> NDArray[np.float64]
```

Calculate per-box, per-species concentration-weighted particle mass.

**Parameters**

- `particles` (`ParticleData`) — Particle data with masses [kg/event] and concentrations [#/m³].

**Returns**

- (`NDArray[np.float64]`) — Particle mass concentration [kg/m³] with shape
- (`NDArray[np.float64]`) — ``(n_boxes, n_species)``.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L903-L920)

## particula.dynamics.nucleation.particle_source.commit_particle_source

*function*

```python
def commit_particle_source(demand: SourceDemandData, diagnostics: SourceDiagnostics, particles: ParticleData, gas: GasData, config: ParticleSourceCommitConfig) -> FinalizedSourceDiagnostics
```

Commit a gas-admitted source into staged fixed-capacity particle storage.

Demand planning event counts are [#/m³]; native commits convert them to
counts at the
current volume. Native representative scaling preserves gas density and
scales existing counts with volume. Legacy commits retain density events
and scale pre-existing gas. Demand planning demand and
final gas removal are mass concentrations [kg/m³], while per-event particle
mass is [kg/event] and ``source_charge`` is in elementary-charge counts.
Policy resolution, resampling, representative-volume scaling, and
activation operate on a private ``ParticleData.copy()``.
A selected representative-volume row scales pre-existing particle and gas
concentrations before final source mass is subtracted. Returned diagnostics
own read-only arrays. Only caller ``masses``, ``concentration``, ``charge``,
``volume``, and gas concentration arrays are mutated, after all validation.

**Parameters**

- `demand` (`SourceDemandData`) — Immutable demand planning per-event mass and provisional gas-demand record.
- `diagnostics` (`SourceDiagnostics`) — Immutable demand planning event-count diagnostics.
- `particles` (`ParticleData`) — Writable fixed-capacity particle data.
- `gas` (`GasData`) — Writable gas concentration inventory.
- `config` (`ParticleSourceCommitConfig`) — Immutable capacity, resampling, and public construction scaling controls.

**Returns**

- (`FinalizedSourceDiagnostics`) — Immutable final event, slot, policy, gas-transfer, and conservation
- (`FinalizedSourceDiagnostics`) — diagnostics.

**Raises**

- `TypeError` — If a boundary object has the wrong type.
- `ValueError` — If schemas, physical values, demand planning records,
capacity policy,
scaling, activation, gas inventory, or conservation are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L923-L1181)

## particula.dynamics.nucleation.particle_source.finalize_particle_source

*function*

```python
def finalize_particle_source(potential_events: PotentialEventData, composition: InjectionComposition, gas: GasData) -> tuple[SourceDemandData, SourceDiagnostics]
```

Finalize immutable, inventory-limited source demand records.

Every box receives one common admitted event count constrained by its
tightest participating gas inventory. The returned records own fresh,
read-only arrays. This concrete demand planning boundary only plans demand:
it neither
mutates inputs or ``gas`` nor accesses particle slots or exhaustion state.

**Parameters**

- `potential_events` (`PotentialEventData`) — Survival-adjusted potential source rate [#/m³/s] and source duration [s]. Survival is not applied again.
- `composition` (`InjectionComposition`) — Nonnegative molecule counts by gas species for each formed event.
- `gas` (`GasData`) — Read-only source inventory with molar masses [kg/mol] and concentrations [kg/m³].

**Returns**

- (`SourceDemandData`) — Immutable per-event mass [kg/event] and provisional gas demand [kg/m³],
- (`SourceDiagnostics`) — followed by event-count diagnostics [#/m³] and limiting species indices.

**Raises**

- `TypeError` — If a top-level input has an invalid type.
- `ValueError` — If schemas, physical values, or derived quantities are
invalid or cannot be made inventory-safe after four ULP corrections.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/particle_source.py#L294-L426)
