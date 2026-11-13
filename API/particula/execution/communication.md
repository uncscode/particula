# particula.execution.communication

*module*

Declare and preflight fixed-shape resident communication maps.

This concrete-only, direct-import boundary performs read-only map validation of
caller-owned Warp map arrays. It neither reads resident primaries, copies map
payloads, nor launches a writer, and is deliberately absent from
``particula.execution`` exports. Empty and all-disabled maps are successful
write-free cases only after complete applicable carrier, schema, alias, and
domain preflight.

Map validation validates topology and rates in 1/s, but receives neither source
inventory
nor time step. Population-dependent outbound-overdraw validation therefore
belongs to gas transport, before its writers launch. Volume evolution owns
volume writers, particle transport owns
Particle transport, and resident binding owns resident binding and
primary/sidecar aliasing.
This module provides none of those operations, synchronization, host
conversion, fallback, or scheduling.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L1-L1)

## particula.execution.communication.CommunicationConfiguration

*class* · *dataclass*

```python
class CommunicationConfiguration
```

Combine immutable communication declarations without binding a session.

The configuration preserves nested carrier and Warp-array identities, has
no session or registry binding, and carries no resident primary state. It
neither reads source population nor accepts a time step, so it cannot
validate population-dependent outbound totals; gas transport must do so
before its
writers run.

**Attributes**

- `communication_map` (`CommunicationMap`) — Fixed-capacity map declaration and caller-owned arrays.
- `prescribed_volume` (`PrescribedVolumeUpdate`) — Optional final-volume declaration; volume evolution owns writes.
- `resource_shapes` (`tuple[CommunicationResourceShape, ...]`) — Unique-role metadata declarations that own no arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L162-L207)

### particula.execution.communication.CommunicationConfiguration.communication_map

*attribute* · *instance attribute*

```python
communication_map: CommunicationMap
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L181-L181)

### particula.execution.communication.CommunicationConfiguration.prescribed_volume

*attribute* · *instance attribute*

```python
prescribed_volume: PrescribedVolumeUpdate
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L182-L182)

### particula.execution.communication.CommunicationConfiguration.resource_shapes

*attribute* · *instance attribute*

```python
resource_shapes: tuple[CommunicationResourceShape, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L183-L183)

### particula.execution.communication.CommunicationConfiguration.__init__

*method*

```python
def __init__(communication_map: CommunicationMap, prescribed_volume: PrescribedVolumeUpdate, resource_shapes: tuple[CommunicationResourceShape, ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L0-L0)

### particula.execution.communication.CommunicationConfiguration.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate cheap nested carrier and unique-role metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L185-L207)

## particula.execution.communication.CommunicationMap

*class* · *dataclass*

```python
class CommunicationMap
```

Retain fixed-capacity caller-owned map arrays by identity.

The frozen carrier keeps the supplied array objects, rather than copies or
normalized replacements. During validation, ``source_boxes``,
``destination_boxes``, and ``enabled`` must be distinct, nonoverlapping,
same-device contiguous pointer-backed ``wp.int32`` arrays of shape
``(E,)``. ``rates`` must be a similarly distinct ``wp.float64`` ``(E,)``
array in 1/s. Validation neither normalizes nor mutates these arrays.

**Attributes**

- `form` (`CommunicationMapForm`) — Enabled-edge topology rule.
- `transport_mode` (`CommunicationTransportMode`) — Payload class that a later phase may transport.
- `edge_capacity` (`int`) — Non-boolean integral edge capacity ``E >= 0``.
- `source_boxes` (`object`) — Caller-owned source-box indices with shape ``(E,)``.
- `destination_boxes` (`object`) — Caller-owned destination-box indices with shape ``(E,)``.
- `enabled` (`object`) — Caller-owned zero-or-one edge flags with shape ``(E,)``.
- `rates` (`object`) — Caller-owned nonnegative finite per-edge rates in 1/s.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L98-L141)

### particula.execution.communication.CommunicationMap.destination_boxes

*attribute* · *instance attribute*

```python
destination_boxes: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L124-L124)

### particula.execution.communication.CommunicationMap.edge_capacity

*attribute* · *instance attribute*

```python
edge_capacity: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L122-L122)

### particula.execution.communication.CommunicationMap.enabled

*attribute* · *instance attribute*

```python
enabled: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L125-L125)

### particula.execution.communication.CommunicationMap.form

*attribute* · *instance attribute*

```python
form: CommunicationMapForm
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L120-L120)

### particula.execution.communication.CommunicationMap.rates

*attribute* · *instance attribute*

```python
rates: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L126-L126)

### particula.execution.communication.CommunicationMap.source_boxes

*attribute* · *instance attribute*

```python
source_boxes: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L123-L123)

### particula.execution.communication.CommunicationMap.transport_mode

*attribute* · *instance attribute*

```python
transport_mode: CommunicationTransportMode
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L121-L121)

### particula.execution.communication.CommunicationMap.__init__

*method*

```python
def __init__(form: CommunicationMapForm, transport_mode: CommunicationTransportMode, edge_capacity: int, source_boxes: object, destination_boxes: object, enabled: object, rates: object) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L0-L0)

### particula.execution.communication.CommunicationMap.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate only fixed-cost carrier metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L128-L141)

## particula.execution.communication.CommunicationMapForm

*class*

```python
class CommunicationMapForm(str, Enum)
```

Bases: `str`, `Enum`

Declare the enabled-edge topology accepted by a communication map.

``ONE_DIMENSIONAL`` permits only neighboring enabled box pairs, while
``ARBITRARY_PAIRS`` permits any distinct in-range directed pair.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L33-L41)

### particula.execution.communication.CommunicationMapForm.ARBITRARY_PAIRS

*attribute* · *class attribute* · *instance attribute*

```python
ARBITRARY_PAIRS = 'arbitrary_pairs'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L41-L41)

### particula.execution.communication.CommunicationMapForm.ONE_DIMENSIONAL

*attribute* · *class attribute* · *instance attribute*

```python
ONE_DIMENSIONAL = 'one_dimensional'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L40-L40)

## particula.execution.communication.CommunicationResourceShape

*class* · *dataclass*

```python
class CommunicationResourceShape
```

Describe one named fixed-shape communication resource.

**Parameters**

- `role` (`str`) — Nonempty, unpadded resource role. This carrier owns no array.
- `dtype` (`object`) — Declared resource dtype metadata, retained unchanged.
- `shape_kind` (`CommunicationShapeKind`) — Exact fixed-shape vocabulary value.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L72-L95)

### particula.execution.communication.CommunicationResourceShape.dtype

*attribute* · *instance attribute*

```python
dtype: object
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L83-L83)

### particula.execution.communication.CommunicationResourceShape.role

*attribute* · *instance attribute*

```python
role: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L82-L82)

### particula.execution.communication.CommunicationResourceShape.shape_kind

*attribute* · *instance attribute*

```python
shape_kind: CommunicationShapeKind
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L84-L84)

### particula.execution.communication.CommunicationResourceShape.__init__

*method*

```python
def __init__(role: str, dtype: object, shape_kind: CommunicationShapeKind) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L0-L0)

### particula.execution.communication.CommunicationResourceShape.__post_init__

*method*

```python
def __post_init__() -> None
```

Perform cheap role and shape metadata validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L86-L95)

## particula.execution.communication.CommunicationShapeKind

*class*

```python
class CommunicationShapeKind(str, Enum)
```

Bases: `str`, `Enum`

Declare fixed resource-shape vocabulary without retaining arrays.

The letters denote fixed dimensions: ``B`` boxes, ``E`` edges, ``S`` gas
species, and ``N`` particle slots.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L57-L69)

### particula.execution.communication.CommunicationShapeKind.B

*attribute* · *class attribute* · *instance attribute*

```python
B = 'b'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L64-L64)

### particula.execution.communication.CommunicationShapeKind.BE

*attribute* · *class attribute* · *instance attribute*

```python
BE = 'be'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L66-L66)

### particula.execution.communication.CommunicationShapeKind.BN

*attribute* · *class attribute* · *instance attribute*

```python
BN = 'bn'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L68-L68)

### particula.execution.communication.CommunicationShapeKind.BNS

*attribute* · *class attribute* · *instance attribute*

```python
BNS = 'bns'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L69-L69)

### particula.execution.communication.CommunicationShapeKind.BS

*attribute* · *class attribute* · *instance attribute*

```python
BS = 'bs'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L67-L67)

### particula.execution.communication.CommunicationShapeKind.E

*attribute* · *class attribute* · *instance attribute*

```python
E = 'e'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L65-L65)

## particula.execution.communication.CommunicationTransportMode

*class*

```python
class CommunicationTransportMode(str, Enum)
```

Bases: `str`, `Enum`

Declare the resident payload class a later phase may transport.

Map validation records this immutable mode but does not read, transport, or
otherwise
validate resident gas or particle payloads.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L44-L54)

### particula.execution.communication.CommunicationTransportMode.GAS

*attribute* · *class attribute* · *instance attribute*

```python
GAS = 'gas'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L52-L52)

### particula.execution.communication.CommunicationTransportMode.GAS_AND_PARTICLES

*attribute* · *class attribute* · *instance attribute*

```python
GAS_AND_PARTICLES = 'gas_and_particles'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L54-L54)

### particula.execution.communication.CommunicationTransportMode.PARTICLES

*attribute* · *class attribute* · *instance attribute*

```python
PARTICLES = 'particles'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L53-L53)

## particula.execution.communication.PrescribedVolumeUpdate

*class* · *dataclass*

```python
class PrescribedVolumeUpdate
```

Retain optional caller-owned final per-box volumes by identity.

``final_volumes`` is retained by identity and is ``None`` or later validates
as a distinct, nonoverlapping, same-device contiguous pointer-backed
``wp.float64`` array shaped ``(B,)`` in m³ with finite positive values. Map
validation
only preflights it; volume evolution owns any volume write.

**Attributes**

- `final_volumes` (`object | None`) — Optional caller-owned final volumes in m³ with shape ``(B,)``.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L144-L159)

### particula.execution.communication.PrescribedVolumeUpdate.final_volumes

*attribute* · *instance attribute*

```python
final_volumes: object | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L159-L159)

### particula.execution.communication.PrescribedVolumeUpdate.__init__

*method*

```python
def __init__(final_volumes: object | None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L0-L0)

## particula.execution.communication._array_range

*function*

```python
def _array_range(value: Any, shape: tuple[int, ...], name: str, item_size: int) -> tuple[int, int] | None
```

Validate contiguous backing metadata and return a nonempty byte range.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L368-L400)

## particula.execution.communication._bitonic_sort_step

*function*

```python
def _bitonic_sort_step(keys: wp.array(dtype=wp.int64), stage: int, offset: int)
```

Execute one compare-exchange stage of an ascending bitonic sort.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L330-L345)

## particula.execution.communication._collect_enabled_keys

*function*

```python
def _collect_enabled_keys(source: wp.array(dtype=wp.int32), destination: wp.array(dtype=wp.int32), enabled: wp.array(dtype=wp.int32), boxes: int, keys: wp.array(dtype=wp.int64), cursor: wp.array(dtype=wp.int32))
```

Collect enabled directed-pair keys into private compact scratch.

The atomic cursor makes the compact order unspecified, which is harmless
because a later deterministic sort compares keys rather than edge order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L307-L327)

## particula.execution.communication._count_enabled

*function*

```python
def _count_enabled(enabled: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32))
```

Count enabled edges into validator-owned scalar storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L296-L304)

## particula.execution.communication._duplicate_scratch_size

*function*

```python
def _duplicate_scratch_size(enabled_edges: int) -> int
```

Return power-of-two private sort storage for enabled edges only.

Disabled fixed-capacity padding never increases this allocation. The
validator uses bounded ``O(E_enabled * log(E_enabled)^2)`` compare-exchange
work, avoiding input-dependent collision probing.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L449-L459)

## particula.execution.communication._is_warp_array

*function*

```python
def _is_warp_array(value: object) -> bool
```

Return whether value is a concrete Warp array carrier.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L360-L365)

## particula.execution.communication._overlaps

*function*

```python
def _overlaps(left: tuple[int, int] | None, right: tuple[int, int] | None) -> bool
```

Return whether two nonempty half-open byte ranges overlap.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L426-L435)

## particula.execution.communication._reject_duplicate_edges

*function*

```python
def _reject_duplicate_edges(source: Any, destination: Any, enabled: Any, boxes: int, edges: int, enabled_edges: int, device: Any) -> None
```

Reject duplicate enabled directed pairs using private sort scratch.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L665-L709)

## particula.execution.communication._reject_payload_aliases

*function*

```python
def _reject_payload_aliases(values: tuple[object | None, ...], ranges: tuple[tuple[int, int] | None, ...]) -> None
```

Reject shared identities and overlapping nonempty payload ranges.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L621-L633)

## particula.execution.communication._scan

*function*

```python
def _scan(value: Any, size: int, kernel: Any, name: str, message: str) -> None
```

Run one read-only scalar-status payload validation scan.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L438-L446)

## particula.execution.communication._scan_enabled

*function*

```python
def _scan_enabled(enabled: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32))
```

Flag any enabled-entry value other than 0 or 1.

**Parameters**

- `enabled` (`wp.array(dtype=wp.int32)`) — One-dimensional integer flag array for enabled edges.
- `invalid` (`wp.array(dtype=wp.int32)`) — Single-element integer status buffer updated in place.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L210-L224)

## particula.execution.communication._scan_rates

*function*

```python
def _scan_rates(rates: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32))
```

Flag any rate that is not finite and nonnegative.

**Parameters**

- `rates` (`wp.array(dtype=wp.float64)`) — One-dimensional floating-point rate array in 1/s.
- `invalid` (`wp.array(dtype=wp.int32)`) — Single-element integer status buffer updated in place.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L227-L241)

## particula.execution.communication._scan_sorted_duplicates

*function*

```python
def _scan_sorted_duplicates(keys: wp.array(dtype=wp.int64), count: int, invalid: wp.array(dtype=wp.int32))
```

Flag adjacent equal valid keys after private sorting.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L348-L357)

## particula.execution.communication._scan_topology

*function*

```python
def _scan_topology(source: wp.array(dtype=wp.int32), destination: wp.array(dtype=wp.int32), enabled: wp.array(dtype=wp.int32), boxes: int, one_dimensional: int, invalid: wp.array(dtype=wp.int32))
```

Flag any enabled edge with invalid bounds or one-dimensional gap.

**Parameters**

- `source` (`wp.array(dtype=wp.int32)`) — One-dimensional integer source-box array.
- `destination` (`wp.array(dtype=wp.int32)`) — One-dimensional integer destination-box array.
- `enabled` (`wp.array(dtype=wp.int32)`) — One-dimensional integer enabled-edge array.
- `boxes` (`int`) — Number of resident boxes; valid enabled endpoints are in ``[0, boxes)``.
- `one_dimensional` (`int`) — Integer flag for the one-dimensional topology rule.
- `invalid` (`wp.array(dtype=wp.int32)`) — Single-element integer status buffer updated in place.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L261-L293)

## particula.execution.communication._scan_volumes

*function*

```python
def _scan_volumes(volumes: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32))
```

Flag any prescribed volume that is not finite and positive.

**Parameters**

- `volumes` (`wp.array(dtype=wp.float64)`) — One-dimensional floating-point final-volume array in m^3.
- `invalid` (`wp.array(dtype=wp.int32)`) — Single-element integer status buffer updated in place.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L244-L258)

## particula.execution.communication._validate_array

*function*

```python
def _validate_array(value: object, shape: tuple[int, ...], dtype: object, device: object, name: str, item_size: int) -> tuple[int, int] | None
```

Validate one fixed-schema Warp array without reading its payload.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L403-L423)

## particula.execution.communication._validate_device

*function*

```python
def _validate_device(device: object) -> None
```

Validate explicit non-null Warp device metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L503-L510)

## particula.execution.communication._validate_dimensions

*function*

```python
def _validate_dimensions(dimensions: object) -> int
```

Validate exact resident dimensions and return the box count.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L491-L500)

## particula.execution.communication._validate_enabled_topology

*function*

```python
def _validate_enabled_topology(map_data: CommunicationMap, boxes: int, device: Any) -> None
```

Validate enabled topology and reject duplicate directed pairs.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L712-L755)

## particula.execution.communication._validate_metadata

*function*

```python
def _validate_metadata(configuration: object, dimensions: object, device: object) -> tuple[CommunicationConfiguration, int]
```

Validate exact outer carriers before inspecting any Warp metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L535-L547)

## particula.execution.communication._validate_nested_metadata

*function*

```python
def _validate_nested_metadata(typed: CommunicationConfiguration) -> None
```

Validate nested carrier, enum, capacity, and resource metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L513-L532)

## particula.execution.communication._validate_payload_arrays

*function*

```python
def _validate_payload_arrays(map_data: CommunicationMap, volumes: object | None, boxes: int, device: Any) -> tuple[tuple[object | None, ...], tuple[tuple[int, int] | None, ...]]
```

Validate payload schemas and return values with their byte ranges.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L550-L618)

## particula.execution.communication._validate_payload_domains

*function*

```python
def _validate_payload_domains(map_data: CommunicationMap, volumes: object | None, boxes: int) -> None
```

Scan enabled flags, rates, and optional volumes in field order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L636-L662)

## particula.execution.communication._validate_resource_shapes

*function*

```python
def _validate_resource_shapes(resource_shapes: object) -> tuple[CommunicationResourceShape, ...]
```

Validate exact resource declarations before payload metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L462-L488)

## particula.execution.communication.validate_communication_configuration

*function*

```python
def validate_communication_configuration(configuration: object, dimensions: object, device: object) -> CommunicationConfiguration
```

Read-only validate one fixed-shape communication configuration.

The validator retains and returns the exact configuration object. It writes
no caller-owned array or carrier and makes no payload copy; private status
and duplicate-detection scratch are permitted. Map index and flag arrays
must be same-device, contiguous, pointer-backed ``wp.int32`` arrays with
shape ``(E,)``; rates must be ``wp.float64`` ``(E,)`` in 1/s. Optional final
volumes must be ``wp.float64`` ``(B,)`` in m³. Array identities and nonempty
half-open byte ranges must not overlap.

**Parameters**

- `configuration` (`object`) — Exact immutable map, optional-volume, and resource-shape carrier.
- `dimensions` (`object`) — Exact resident dimensions; only ``n_boxes`` (``B``) is used.
- `device` (`object`) — Explicit active Warp device metadata for all payload arrays.

**Returns**

- (`CommunicationConfiguration`) — The exact unchanged ``configuration`` object.

**Raises**

- `TypeError` — If outer carriers or device metadata are invalid.
- `ValueError` — If array schema, identity ownership, byte-range aliasing,
value domains, or enabled topology is invalid.

> **Notes**
>
> Empty and all-disabled maps still receive full applicable preflight and
> are write-free on success. This validates map topology and rate domains
> only. It has neither source-inventory nor time-step input and cannot
> validate outbound population totals; gas transport must enforce that
> condition
> before writers run.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/communication.py#L758-L805)
