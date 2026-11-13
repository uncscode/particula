# particula.gpu.kernels.communication

*module*

Apply concrete-only direct-Warp volume, gas, and particle communication.

The direct kernels preserve extensive inventories during prescribed per-box
volume changes, synchronously transport gas with caller-owned ledgers, and
move fixed-capacity particle populations with caller-owned planning buffers.
``ParticleCommunicationBuffers`` and ``particle_communication_step_gpu`` are
available only from ``particula.gpu.kernels.communication``; neither name is
exported by ``particula.gpu.kernels``, ``particula.gpu``, or the top-level
package. The module performs no hidden host transfer, synchronization,
resizing, conversion, or CPU fallback. Callers synchronize explicitly before
inspecting asynchronous device results.

Private prepared calls preserve setup validation and fallback allocation in
setup, then enqueue retained device work without validation, allocation,
readback, transfer, synchronization, lookup, rebinding, or recovery. Empty or
no-work calls are write-free. Rejection is pre-writer; rollback is not promised
after a writer launches.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L1-L1)

## particula.gpu.kernels.communication._PARTICLE_COMMUNICATION_MAX_PLAN_WORK

*attribute* · *module attribute*

```python
_PARTICLE_COMMUNICATION_MAX_PLAN_WORK = 10000000
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L54-L54)

## particula.gpu.kernels.communication.GasCommunicationBuffers

*class* · *dataclass*

```python
class GasCommunicationBuffers
```

Caller-owned gas-transport work and open-boundary ledgers.

All arrays are contiguous active-device ``wp.float64`` arrays shaped
``(B, S)``. ``amounts``, ``amount_deltas``, and ``outbound_amounts`` are
required work storage. ``source_amounts`` and ``sink_amounts`` are the
optional accounting ledgers required for enabled maps containing,
respectively, a ``-1`` source or destination endpoint. The communication
step overwrites applicable work and accounting ledgers; callers retain
ownership and synchronize before inspection.

**Attributes**

- `amounts` (`Any`) — Immutable per-step gas amount ledger.
- `amount_deltas` (`Any`) — Net in-domain amount change for each box and species.
- `outbound_amounts` (`Any`) — Aggregate amount debited from each in-domain source.
- `source_amounts` (`Any | None`) — Optional positive amount admitted at each destination from declared open sources.
- `sink_amounts` (`Any | None`) — Optional positive amount removed from each source by declared open sinks.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L57-L83)

### particula.gpu.kernels.communication.GasCommunicationBuffers.amount_deltas

*attribute* · *instance attribute*

```python
amount_deltas: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L80-L80)

### particula.gpu.kernels.communication.GasCommunicationBuffers.amounts

*attribute* · *instance attribute*

```python
amounts: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L79-L79)

### particula.gpu.kernels.communication.GasCommunicationBuffers.outbound_amounts

*attribute* · *instance attribute*

```python
outbound_amounts: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L81-L81)

### particula.gpu.kernels.communication.GasCommunicationBuffers.sink_amounts

*attribute* · *class attribute* · *instance attribute*

```python
sink_amounts: Any | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L83-L83)

### particula.gpu.kernels.communication.GasCommunicationBuffers.source_amounts

*attribute* · *class attribute* · *instance attribute*

```python
source_amounts: Any | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L82-L82)

### particula.gpu.kernels.communication.GasCommunicationBuffers.__init__

*method*

```python
def __init__(amounts: Any, amount_deltas: Any, outbound_amounts: Any, source_amounts: Any | None = None, sink_amounts: Any | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L0-L0)

## particula.gpu.kernels.communication.ParticleCommunicationBuffers

*class* · *dataclass*

```python
class ParticleCommunicationBuffers
```

Own planning arrays for concrete-only fixed-capacity particle transport.

Import this carrier only from ``particula.gpu.kernels.communication``; it
is deliberately not exported by ``particula.gpu.kernels``,
``particula.gpu``, or the top-level package. All arrays must be contiguous,
active-device Warp arrays. Debit and credit ledgers use ``wp.float64`` and
shape ``(B, N)``; assignments use ``wp.int32`` and requests use
``wp.float64``, both with shape ``(E, N)``. Successful nonzero planning
overwrites these arrays, while valid no-op calls leave them untouched.

**Attributes**

- `source_debits` (`Any`) — ``wp.float64`` aggregate source debits shaped ``(B, N)``.
- `destination_credits` (`Any`) — ``wp.float64`` aggregate destination credits shaped ``(B, N)``.
- `assignments` (`Any`) — ``wp.int32`` planned destination slots shaped ``(E, N)``.
- `request_concentrations` (`Any`) — ``wp.float64`` immutable-pre-step requests shaped ``(E, N)``.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L86-L110)

### particula.gpu.kernels.communication.ParticleCommunicationBuffers.assignments

*attribute* · *instance attribute*

```python
assignments: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L109-L109)

### particula.gpu.kernels.communication.ParticleCommunicationBuffers.destination_credits

*attribute* · *instance attribute*

```python
destination_credits: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L108-L108)

### particula.gpu.kernels.communication.ParticleCommunicationBuffers.request_concentrations

*attribute* · *instance attribute*

```python
request_concentrations: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L110-L110)

### particula.gpu.kernels.communication.ParticleCommunicationBuffers.source_debits

*attribute* · *instance attribute*

```python
source_debits: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L107-L107)

### particula.gpu.kernels.communication.ParticleCommunicationBuffers.__init__

*method*

```python
def __init__(source_debits: Any, destination_credits: Any, assignments: Any, request_concentrations: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L0-L0)

## particula.gpu.kernels.communication._apply_resident_scaled_concentration

*function*

```python
def _apply_resident_scaled_concentration(concentration: wp.array2d(dtype=wp.float64), old_volumes: wp.array(dtype=wp.float64), final_volumes: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Scale only changed resident rows, preserving equal-volume no-op lanes.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L1883-L1895)

## particula.gpu.kernels.communication._apply_resident_volume_evolution

*function*

```python
def _apply_resident_volume_evolution(volume: wp.array(dtype=wp.float64), final_volumes: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Write only changed resident volume lanes after the bound status scan.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L1871-L1880)

## particula.gpu.kernels.communication._apply_scaled_concentration

*function*

```python
def _apply_scaled_concentration(concentration: wp.array2d(dtype=wp.float64), old_volumes: wp.array(dtype=wp.float64), final_volumes: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32), changed: wp.array(dtype=wp.int32)) -> None
```

Apply validated per-box concentration scaling.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L398-L410)

## particula.gpu.kernels.communication._apply_volume_evolution

*function*

```python
def _apply_volume_evolution(volume: wp.array(dtype=wp.float64), final_volumes: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32), changed: wp.array(dtype=wp.int32)) -> None
```

Write validated final volume values.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L385-L395)

## particula.gpu.kernels.communication._array_range

*function*

```python
def _array_range(array: Any, name: str) -> tuple[int, int] | None
```

Validate pointer-backed contiguous capacity and return its byte range.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L421-L458)

## particula.gpu.kernels.communication._communication_clear_and_stage

*function*

```python
def _communication_clear_and_stage(concentration: wp.array2d(dtype=wp.float64), volume: wp.array(dtype=wp.float64), amounts: wp.array2d(dtype=wp.float64), deltas: wp.array2d(dtype=wp.float64), outbound: wp.array2d(dtype=wp.float64), source_ledger: wp.array2d(dtype=wp.float64), sink_ledger: wp.array2d(dtype=wp.float64), has_source_ledger: int, has_sink_ledger: int, active: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32)) -> None
```

Stage immutable extensive amounts and clear documented work ledgers.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L195-L221)

## particula.gpu.kernels.communication._communication_commit

*function*

```python
def _communication_commit(concentration: wp.array2d(dtype=wp.float64), volume: wp.array(dtype=wp.float64), amounts: wp.array2d(dtype=wp.float64), deltas: wp.array2d(dtype=wp.float64), active: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32)) -> None
```

Commit the validated gas concentration exactly once.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L292-L306)

## particula.gpu.kernels.communication._communication_propose

*function*

```python
def _communication_propose(source: wp.array(dtype=wp.int32), destination: wp.array(dtype=wp.int32), enabled: wp.array(dtype=wp.int32), rates: wp.array(dtype=wp.float64), time_step: wp.float64, amounts: wp.array2d(dtype=wp.float64), deltas: wp.array2d(dtype=wp.float64), outbound: wp.array2d(dtype=wp.float64), source_ledger: wp.array2d(dtype=wp.float64), sink_ledger: wp.array2d(dtype=wp.float64), has_source_ledger: int, has_sink_ledger: int, active: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32)) -> None
```

Aggregate synchronous transfers from the immutable amount ledger.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L224-L265)

## particula.gpu.kernels.communication._communication_validate_final

*function*

```python
def _communication_validate_final(amounts: wp.array2d(dtype=wp.float64), deltas: wp.array2d(dtype=wp.float64), outbound: wp.array2d(dtype=wp.float64), volume: wp.array(dtype=wp.float64), active: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32)) -> None
```

Reject aggregate overdraw and invalid final normalization values.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L268-L289)

## particula.gpu.kernels.communication._communication_validate_final_volumes

*function*

```python
def _communication_validate_final_volumes(final_volumes: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Flag invalid read-only prescribed-volume metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L130-L138)

## particula.gpu.kernels.communication._communication_validate_map

*function*

```python
def _communication_validate_map(source: wp.array(dtype=wp.int32), destination: wp.array(dtype=wp.int32), enabled: wp.array(dtype=wp.int32), rates: wp.array(dtype=wp.float64), boxes: int, one_dimensional: int, has_source_ledger: int, has_sink_ledger: int, active: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32)) -> None
```

Validate enabled edges, including the gas transport ``-1`` boundary
sentinel.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L141-L192)

## particula.gpu.kernels.communication._communication_validate_primaries

*function*

```python
def _communication_validate_primaries(volume: wp.array(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Flag invalid volume or gas concentration values.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L113-L127)

## particula.gpu.kernels.communication._enqueue_prepared_resident_gas_communication

*function*

```python
def _enqueue_prepared_resident_gas_communication(volume: Any, concentration: Any, boxes: int, species: int, source_boxes: Any, destination_boxes: Any, enabled: Any, rates: Any, edge_capacity: int, time_step: float, device: Any, buffers: GasCommunicationBuffers, invalid: Any, active: Any) -> None
```

Launch already-bound GAS communication without validation or allocation.

Caller-owned arrays, map arrays, status arrays, and ledgers remain
resident and are passed directly to the established staged GAS kernels.
The helper performs no setup work, host inspection, lookup, rebinding, or
synchronization; empty/no-work inputs are write-free. A writer launch has
no rollback guarantee.

**Parameters**

- `volume` (`Any`) — Bound resident per-box volume array.
- `concentration` (`Any`) — Bound resident gas concentration array.
- `boxes` (`int`) — Setup-frozen resident box count.
- `species` (`int`) — Setup-frozen resident gas species count.
- `source_boxes` (`Any`) — Bound source endpoint array.
- `destination_boxes` (`Any`) — Bound destination endpoint array.
- `enabled` (`Any`) — Bound edge-enable array.
- `rates` (`Any`) — Bound edge-rate array.
- `edge_capacity` (`int`) — Number of retained edge slots.
- `time_step` (`float`) — Validated barrier duration in seconds.
- `device` (`Any`) — Device hosting all bound arrays.
- `buffers` (`GasCommunicationBuffers`) — Pinned GAS work ledgers.
- `invalid` (`Any`) — Pinned validation status array.
- `active` (`Any`) — Pinned activity status array.

**Returns**

- (`None`) — None. The caller retains the bound arrays by identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L1898-L2014)

## particula.gpu.kernels.communication._enqueue_prepared_resident_particle_communication

*function*

```python
def _enqueue_prepared_resident_particle_communication(masses: Any, concentration: Any, charge: Any, density: Any, volume: Any, boxes: int, slots: int, species: int, source_boxes: Any, destination_boxes: Any, enabled: Any, rates: Any, edge_capacity: int, one_dimensional: int, time_step: float, device: Any, buffers: ParticleCommunicationBuffers, invalid: Any, demand: Any, initial_masses: Any, initial_concentration: Any, initial_charge: Any) -> None
```

Launch bound PARTICLES communication without setup or allocation.

The planner, snapshot, and gated commit run against caller-owned resident
arrays. This seam performs no validation, lookup, rebinding, transfer,
synchronization, or host inspection. Empty/no-work inputs are write-free;
rollback is not promised after a writer launches.

**Parameters**

- `masses` (`Any`) — Bound resident particle mass array.
- `concentration` (`Any`) — Bound resident particle concentration array.
- `charge` (`Any`) — Bound resident particle charge array.
- `density` (`Any`) — Bound resident particle density array.
- `volume` (`Any`) — Bound resident per-box volume array.
- `boxes` (`int`) — Setup-frozen resident box count.
- `slots` (`int`) — Setup-frozen particle capacity.
- `species` (`int`) — Setup-frozen particle species count.
- `source_boxes` (`Any`) — Bound source endpoint array.
- `destination_boxes` (`Any`) — Bound destination endpoint array.
- `enabled` (`Any`) — Bound edge-enable array.
- `rates` (`Any`) — Bound edge-rate array.
- `edge_capacity` (`int`) — Number of retained edge slots.
- `one_dimensional` (`int`) — Integer flag for one-dimensional map validation.
- `time_step` (`float`) — Validated barrier duration in seconds.
- `device` (`Any`) — Device hosting all bound arrays.
- `buffers` (`ParticleCommunicationBuffers`) — Pinned PARTICLES planning buffers.
- `invalid` (`Any`) — Pinned validation status array.
- `demand` (`Any`) — Pinned demand status array.
- `initial_masses` (`Any`) — Pinned immutable mass snapshot.
- `initial_concentration` (`Any`) — Pinned immutable concentration snapshot.
- `initial_charge` (`Any`) — Pinned immutable charge snapshot.

**Returns**

- (`None`) — None. The caller retains the bound arrays by identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L2017-L2148)

## particula.gpu.kernels.communication._enqueue_prepared_resident_volume_evolution

*function*

```python
def _enqueue_prepared_resident_volume_evolution(volume: Any, particle_concentration: Any, gas_concentration: Any, boxes: int, final_volumes: Any, invalid: Any, changed: Any, device: Any) -> None
```

Launch a bound volume barrier with an equal-volume write-free path.

The device-resident status scan leaves status lanes and primary arrays
untouched when every final volume equals its current value. Changed values
use the established concentration scaling and volume writer. This retained
seam does no host inspection, lookup, rebinding, or recovery; rollback is
not promised after a writer launch.

**Parameters**

- `volume` (`Any`) — Bound resident per-box volume array.
- `particle_concentration` (`Any`) — Bound resident particle concentration array.
- `gas_concentration` (`Any`) — Bound resident gas concentration array.
- `boxes` (`int`) — Setup-frozen resident box count.
- `final_volumes` (`Any`) — Caller-owned bound final volumes in m^3.
- `invalid` (`Any`) — Pinned volume validation status array.
- `changed` (`Any`) — Pinned volume-change status array.
- `device` (`Any`) — Device hosting all bound arrays.

**Returns**

- (`None`) — None. The caller retains the bound arrays by identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L2151-L2224)

## particula.gpu.kernels.communication._get_field

*function*

```python
def _get_field(container: Any, field: str, name: str) -> Any
```

Return a required field with a stable malformed-container error.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L413-L418)

## particula.gpu.kernels.communication._particle_communication_commit

*function*

```python
def _particle_communication_commit(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), initial_masses: wp.array3d(dtype=wp.float64), initial_concentration: wp.array2d(dtype=wp.float64), initial_charge: wp.array2d(dtype=wp.float64), source: wp.array(dtype=wp.int32), destination: wp.array(dtype=wp.int32), debits: wp.array2d(dtype=wp.float64), credits: wp.array2d(dtype=wp.float64), assignments: wp.array2d(dtype=wp.int32), invalid: wp.array(dtype=wp.int32), demand: wp.array(dtype=wp.int32)) -> None
```

Commit a fully validated plan in one gated primary writer.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L1137-L1178)

## particula.gpu.kernels.communication._particle_communication_plan

*function*

```python
def _particle_communication_plan(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), density: wp.array(dtype=wp.float64), volume: wp.array(dtype=wp.float64), source: wp.array(dtype=wp.int32), destination: wp.array(dtype=wp.int32), enabled: wp.array(dtype=wp.int32), rates: wp.array(dtype=wp.float64), one_dimensional: int, time_step: wp.float64, debits: wp.array2d(dtype=wp.float64), credits: wp.array2d(dtype=wp.float64), assignments: wp.array2d(dtype=wp.int32), requests: wp.array2d(dtype=wp.float64), invalid: wp.array(dtype=wp.int32), demand: wp.array(dtype=wp.int32)) -> None
```

Plan deterministically, writing public buffers only after demand.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L923-L1134)

## particula.gpu.kernels.communication._particle_key_equal

*function*

```python
def _particle_key_equal(left_masses: wp.array3d(dtype=wp.float64), left_box: int, left_slot: int, right_masses: wp.array3d(dtype=wp.float64), right_box: int, right_slot: int, charge: wp.array2d(dtype=wp.float64), species: int) -> bool
```

Return equality; floating equality canonicalizes signed zero.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L900-L920)

## particula.gpu.kernels.communication._reject_aliases

*function*

```python
def _reject_aliases(arrays: tuple[Any, ...], ranges: tuple[tuple[int, int] | None, ...]) -> None
```

Reject shared identities and nonempty overlapping primary storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L483-L500)

## particula.gpu.kernels.communication._resident_prepare_volume_status

*function*

```python
def _resident_prepare_volume_status(volume: wp.array(dtype=wp.float64), final_volumes: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32), changed: wp.array(dtype=wp.int32), boxes: int) -> None
```

Reset resident volume statuses only when a final volume differs.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L1854-L1868)

## particula.gpu.kernels.communication._resident_reset_status

*function*

```python
def _resident_reset_status(invalid: wp.array(dtype=wp.int32), active: wp.array(dtype=wp.int32), active_value: int) -> None
```

Reset registry-pinned one-element status storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L1842-L1851)

## particula.gpu.kernels.communication._resolved_communication

*function*

```python
def _resolved_communication(particles: Any) -> None
```

Require explicit resolved-count population metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L542-L546)

## particula.gpu.kernels.communication._safe_byte_range

*function*

```python
def _safe_byte_range(pointer: int, count: int, item_size: int, name: str) -> tuple[int, int]
```

Return a validated inclusive-exclusive byte range for storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L339-L348)

## particula.gpu.kernels.communication._safe_dimension_product

*function*

```python
def _safe_dimension_product(lengths: tuple[int, ...], name: str) -> int
```

Return the element count for a shape or raise on overflow.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L329-L336)

## particula.gpu.kernels.communication._scan_1d

*function*

```python
def _scan_1d(values: Any, kernel: Any, invalid: Any) -> None
```

Launch a device-resident one-dimensional validation scan.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L503-L512)

## particula.gpu.kernels.communication._scan_2d

*function*

```python
def _scan_2d(values: Any, old_volumes: Any, final_volumes: Any, invalid: Any, changed: Any, product: bool) -> None
```

Launch a device-resident concentration validation scan.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L515-L539)

## particula.gpu.kernels.communication._scan_nonnegative_finite

*function*

```python
def _scan_nonnegative_finite(values: wp.array2d(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Flag values that are not finite and nonnegative.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L319-L326)

## particula.gpu.kernels.communication._scan_positive_finite

*function*

```python
def _scan_positive_finite(values: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Flag values that are not finite and positive.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L309-L316)

## particula.gpu.kernels.communication._snapshot_particle_communication_fields

*function*

```python
def _snapshot_particle_communication_fields(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), initial_masses: wp.array3d(dtype=wp.float64), initial_concentration: wp.array2d(dtype=wp.float64), initial_charge: wp.array2d(dtype=wp.float64), invalid: wp.array(dtype=wp.int32), demand: wp.array(dtype=wp.int32)) -> None
```

Copy the immutable particle fields used by the gated commit writer.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L1181-L1199)

## particula.gpu.kernels.communication._validate_array

*function*

```python
def _validate_array(value: Any, name: str, dtype: Any, rank: int, shape: tuple[int, ...] | None = None, device: Any | None = None) -> tuple[Any, tuple[int, int] | None]
```

Validate one primary array's fixed schema and memory backing.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L461-L480)

## particula.gpu.kernels.communication._validate_count_density

*function*

```python
def _validate_count_density(counts: wp.array2d(dtype=wp.float64), volumes: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Gate unrepresentable number densities before physical volume writes.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L549-L563)

## particula.gpu.kernels.communication._validate_factors

*function*

```python
def _validate_factors(old_volumes: wp.array(dtype=wp.float64), final_volumes: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32), changed: wp.array(dtype=wp.int32)) -> None
```

Validate factors and record whether any volume differs.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L351-L364)

## particula.gpu.kernels.communication._validate_gas_communication_configuration

*function*

```python
def _validate_gas_communication_configuration(configuration: object) -> CommunicationConfiguration
```

Exact-check the map validation declaration carriers needed by the gas
transport boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L832-L863)

## particula.gpu.kernels.communication._validate_particle_communication_configuration

*function*

```python
def _validate_particle_communication_configuration(configuration: object) -> CommunicationConfiguration
```

Exact-check the map validation declaration carriers for particle
transport.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L866-L897)

## particula.gpu.kernels.communication._validate_resource_shapes

*function*

```python
def _validate_resource_shapes(resource_shapes: object) -> None
```

Validate the exact map validation resource-shape declaration metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L808-L829)

## particula.gpu.kernels.communication._validate_scaled_concentration

*function*

```python
def _validate_scaled_concentration(concentration: wp.array2d(dtype=wp.float64), old_volumes: wp.array(dtype=wp.float64), final_volumes: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32), changed: wp.array(dtype=wp.int32)) -> None
```

Reject overflowing or positive-to-zero scaled concentrations.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L367-L382)

## particula.gpu.kernels.communication._validate_time_step

*function*

```python
def _validate_time_step(time_step: object) -> float
```

Validate a nonnegative finite scalar time in seconds before access.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L796-L805)

## particula.gpu.kernels.communication.gas_communication_step_gpu

*function*

```python
def gas_communication_step_gpu(particles: Any, gas: Any, configuration: object, time_step: object, amounts: object | GasCommunicationBuffers, amount_deltas: object | None = None, outbound_amounts: object | None = None, source_amounts: object | None = None, sink_amounts: object | None = None) -> tuple[Any, Any]
```

Synchronously mix gas with caller-owned extensive amount ledgers.

The immutable staged ledger is ``amounts = concentration * volume`` in
amount units. An in-domain ``source -> destination`` edge transfers
``amounts[source] * rate * time_step``, debiting and crediting
``amount_deltas``. An open sink ``source -> -1`` uses that same source
amount, debits ``amount_deltas``, and records a positive ``sink_amounts``
entry. An open source ``-1 -> destination`` instead uses the destination
box's pre-step amount and records its positive transfer in
``source_amounts``. The sole primary writer assigns
``gas.concentration = (amounts + amount_deltas) / volume``.

``amounts``, ``amount_deltas``, and ``outbound_amounts`` are caller-owned
contiguous ``wp.float64`` arrays shaped ``(B, S)``. A
``GasCommunicationBuffers`` carrier may provide them. Open accounting
ledgers have the same schema and are required for enabled edges with their
respective boundary direction. Closed maps conserve each species' extensive
amount up to atomic floating-point tolerance. Particle fields, prescribed
volumes, and map metadata are read-only. Valid zero-size, empty-map, and
zero-time calls validate observable schemas then leave all buffers and
primaries unchanged.

The direct call performs no hidden transfer, synchronization, conversion,
resize, CPU fallback, or post-launch rollback. It has device work of
``O(E * S + B * S)`` for ``E`` edge slots, ``B`` boxes, and ``S`` gas
species. Callers synchronize before inspecting results.

**Parameters**

- `particles` (`Any`) — Complete caller-owned Warp particle container.
- `gas` (`Any`) — Complete caller-owned Warp gas container.
- `configuration` (`object`) — Exact map validation gas ``CommunicationConfiguration``.
- `time_step` (`object`) — Finite nonnegative explicit-Euler step in seconds.
- `amounts` (`object | GasCommunicationBuffers`) — Work ledger or concrete-only ``GasCommunicationBuffers`` carrier.
- `amount_deltas` (`object | None`) (default: `None`) — Caller-owned net-delta work ledger when no carrier is provided.
- `outbound_amounts` (`object | None`) (default: `None`) — Caller-owned source-debit work ledger when no carrier is provided.
- `source_amounts` (`object | None`) (default: `None`) — Optional open-source accounting ledger.
- `sink_amounts` (`object | None`) (default: `None`) — Optional open-sink accounting ledger.

**Returns**

- (`tuple[Any, Any]`) — The exact input ``(particles, gas)`` objects.

**Raises**

- `TypeError` — If scalar or declaration carrier types are invalid.
- `ValueError` — If observable array schemas, devices, aliases, buffer
requirements, or transport declaration are invalid. Device-domain
failures gate primary writes.

> **Note**
>
> Applicable work and accounting ledgers may be written before an
> asynchronous device-domain failure. Rollback is not promised after the
> primary commit kernel launches.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L1202-L1560)

## particula.gpu.kernels.communication.particle_communication_step_gpu

*function*

```python
def particle_communication_step_gpu(particles: Any, configuration: object, time_step: object, buffers: object) -> Any
```

Transport fixed-capacity particle populations across a closed map.

This concrete-only API is available only from
``particula.gpu.kernels.communication`` and is deliberately not exported
by ``particula.gpu.kernels``, ``particula.gpu``, or the top-level package.
Immutable pre-step particle fields determine requests. The planner
deterministically prefers an exact population match, including signed-zero
equivalence, then reserves the next ascending pre-step free destination
slot. Population species-mass lanes and signed charge move together. The
primary arrays are contiguous active-device ``wp.float64`` arrays: masses
have shape ``(B, N, S)``, concentration and charge have shape ``(B, N)``,
density has shape ``(S,)``, and volume has shape ``(B,)``. A slot must be
active (positive finite concentration, finite nonnegative mass lanes with
positive total mass, and finite charge) or exactly free (all-zero fields).
``buffers`` owns float64 debit/credit ``(B, N)`` arrays, int32 assignments
``(E, N)``, and float64 requests ``(E, N)``. Here ``E`` is map edge
capacity. Each source count debit is credited unchanged at the destination,
preserving extensive inventories across unequal box volumes.

Valid zero-size, zero-time, disabled-map, and zero-demand calls are
write-free, including all supplied buffers. A nonzero plan overwrites the
documented buffers. Host/schema preflight rejects without caller mutation;
device-domain failures gate the single primary commit, although planning
buffers may already have changed. Rollback is not promised after the commit
writer launches. Closed-map transport conserves concentration-weighted
particle number, every species-mass lane, and signed charge. Callers own
device placement and explicit Warp synchronization before inspection; this
boundary does not transfer, resize, compact, implicitly activate slots, or
provide a CPU fallback.

Native resolved populations carry counts. Physical volume evolution keeps
those counts unchanged; transport moves equal counts between boxes. Native
PMF, PDF and untagged populations are unsupported.

**Parameters**

- `particles` (`Any`) — Complete caller-owned fixed-shape Warp particle container.
- `configuration` (`object`) — Exact map validation particle ``CommunicationConfiguration``.
- `time_step` (`object`) — Finite nonnegative explicit-Euler time step in seconds.
- `buffers` (`object`) — Exact concrete ``ParticleCommunicationBuffers`` carrier.

**Returns**

- (`Any`) — The exact input ``particles`` object.

**Raises**

- `TypeError` — If scalar, declaration, or buffer-carrier types are invalid.
- `ValueError` — If observable schemas, devices, aliases, or declaration
metadata are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L1563-L1839)

## particula.gpu.kernels.communication.resident_gas_communication_step_gpu

*function*

```python
def resident_gas_communication_step_gpu(particles: Any, gas: Any, configuration: CommunicationConfiguration, time_step: float, buffers: GasCommunicationBuffers, invalid: Any, active: Any) -> tuple[Any, Any]
```

Dispatch acquired closed GAS resources without public map validation.

This private resident seam assumes acquisition has already established every
schema, identity, alias, and static map invariant. It intentionally has no
allocations, payload scans, host readback, or synchronization.

**Parameters**

- `particles` (`Any`) — Acquired resident particle container.
- `gas` (`Any`) — Acquired resident gas container.
- `configuration` (`CommunicationConfiguration`) — Acquired closed GAS communication configuration.
- `time_step` (`float`) — Validated nonnegative barrier duration in seconds.
- `buffers` (`GasCommunicationBuffers`) — Registry-pinned GAS work buffers.
- `invalid` (`Any`) — Registry-pinned validation status array.
- `active` (`Any`) — Registry-pinned activity status array.

**Returns**

- (`tuple[Any, Any]`) — The exact ``(particles, gas)`` input containers.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L2227-L2271)

## particula.gpu.kernels.communication.resident_particle_communication_step_gpu

*function*

```python
def resident_particle_communication_step_gpu(particles: Any, configuration: CommunicationConfiguration, time_step: float, buffers: ParticleCommunicationBuffers, invalid: Any, demand: Any, initial_masses: Any, initial_concentration: Any, initial_charge: Any) -> Any
```

Dispatch acquired closed PARTICLES resources without public map
    validation checks.

Acquisition owns payload validation and allocation. This resident-only seam
uses the registry-pinned planner, status, and snapshot storage by identity.

**Parameters**

- `particles` (`Any`) — Acquired resident particle container.
- `configuration` (`CommunicationConfiguration`) — Acquired closed PARTICLES communication configuration.
- `time_step` (`float`) — Validated nonnegative barrier duration in seconds.
- `buffers` (`ParticleCommunicationBuffers`) — Registry-pinned PARTICLES planning buffers.
- `invalid` (`Any`) — Registry-pinned validation status array.
- `demand` (`Any`) — Registry-pinned demand status array.
- `initial_masses` (`Any`) — Registry-pinned particle mass snapshot.
- `initial_concentration` (`Any`) — Registry-pinned concentration snapshot.
- `initial_charge` (`Any`) — Registry-pinned charge snapshot.

**Returns**

- (`Any`) — The exact ``particles`` input container.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L2274-L2331)

## particula.gpu.kernels.communication.resident_volume_evolution_step_gpu

*function*

```python
def resident_volume_evolution_step_gpu(particles: Any, gas: Any, final_volumes: Any, invalid: Any, changed: Any) -> tuple[Any, Any]
```

Apply acquired prescribed volumes without public map validation.

**Parameters**

- `particles` (`Any`) — Acquired resident particle container.
- `gas` (`Any`) — Acquired resident gas container.
- `final_volumes` (`Any`) — Registry-pinned prescribed per-box volumes in m^3.
- `invalid` (`Any`) — Registry-pinned volume validation status array.
- `changed` (`Any`) — Registry-pinned volume-change status array.

**Returns**

- (`tuple[Any, Any]`) — The exact ``(particles, gas)`` input containers.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L2334-L2364)

## particula.gpu.kernels.communication.volume_evolution_step_gpu

*function*

```python
def volume_evolution_step_gpu(particles: Any, gas: Any, final_volumes: object) -> tuple[Any, Any]
```

Apply prescribed final volumes while preserving extensive inventories.

``final_volumes`` must be a caller-owned, contiguous active-device
``wp.float64`` array shaped ``(B,)`` in m^3. Complete read-only preflight
validates schemas and launches device-resident domain, factor, and proposed
product checks before gated writers. Invalid device values suppress every
writer; errors from asynchronous device execution are observed by the caller
at its explicit synchronization boundary. Successful calls update only
``particles.volume`` and scale gas concentrations by the ratio
``old_volume / final_volume``.
Particle counts remain unchanged. Equal final volumes are write-free no-ops.
This direct kernel performs no hidden transfer, synchronization, or CPU
fallback; callers synchronize before inspection.

Native resolved populations carry counts. Physical volume evolution keeps
those counts unchanged; transport moves equal counts between boxes. Native
PMF, PDF and untagged populations are unsupported.

**Parameters**

- `particles` (`Any`) — Warp particle container with fixed-shape primary fields.
- `gas` (`Any`) — Warp gas container with fixed-shape primary fields sharing the particle box count and device.
- `final_volumes` (`object`) — Caller-owned active-device ``wp.float64`` array of final per-box volumes in m^3 with shape ``(B,)``.

**Returns**

- (`Any`) — The exact input ``(particles, gas)`` containers after an in-place update
- (`Any`) — or unchanged after a valid equal-volume no-op.

**Raises**

- `ValueError` — If host-observable schemas, devices, or storage ownership
are invalid. Invalid device values suppress writers and leave
caller-owned inputs unchanged.

> **Note**
>
> Rollback is not promised if an asynchronous apply writer fails after it
> launches.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/communication.py#L566-L793)
