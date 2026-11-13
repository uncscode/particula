# particula.gpu.kernels.slot_management

*module*

Classify and activate fixed-capacity particle slots on a Warp device.

This module provides concrete-module-only GPU diagnostics diagnostics and the
package-exported GPU activation ``activate_slots_gpu`` boundary. Both APIs read
only
particle masses, concentration, and charge; density and volume are
intentionally unobserved. GPU activation validates all metadata, ownership,
current slot
State, counts, capacity, and selected request prefixes before launching its
writer, then writes caller-owned activation and diagnostic sidecars. Caller-
owned simulation arrays are never implicitly transferred; validation may
synchronize and read back private scalar status buffers.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L1-L1)

## particula.gpu.kernels.slot_management._ALIAS_MESSAGE

*attribute* · *module attribute*

```python
_ALIAS_MESSAGE = 'Activation arrays must not share storage.'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L40-L40)

## particula.gpu.kernels.slot_management._CAPACITY_MESSAGE

*attribute* · *module attribute*

```python
_CAPACITY_MESSAGE = 'Requested activation exceeds free slot capacity.'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L38-L38)

## particula.gpu.kernels.slot_management._CATEGORY_ACTIVE

*attribute* · *module attribute*

```python
_CATEGORY_ACTIVE = wp.int32(1)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L33-L33)

## particula.gpu.kernels.slot_management._CATEGORY_FREE

*attribute* · *module attribute*

```python
_CATEGORY_FREE = wp.int32(2)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L34-L34)

## particula.gpu.kernels.slot_management._CATEGORY_INVALID

*attribute* · *module attribute*

```python
_CATEGORY_INVALID = wp.int32(3)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L35-L35)

## particula.gpu.kernels.slot_management._INVALID_COUNT_MESSAGE

*attribute* · *module attribute*

```python
_INVALID_COUNT_MESSAGE = 'requested_counts exceeds request capacity.'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L37-L37)

## particula.gpu.kernels.slot_management._INVALID_REQUEST_MESSAGE

*attribute* · *module attribute*

```python
_INVALID_REQUEST_MESSAGE = 'Requested activation record is invalid.'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L39-L39)

## particula.gpu.kernels.slot_management._INVALID_SLOT_MESSAGE

*attribute* · *module attribute*

```python
_INVALID_SLOT_MESSAGE = 'Invalid particle slot state.'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L36-L36)

## particula.gpu.kernels.slot_management._activate_nonempty_requests

*function*

```python
def _activate_nonempty_requests(masses: Any, concentration: Any, charge: Any, request_masses: Any, request_concentration: Any, request_charge: Any, requested_counts: Any, activated_counts: Any, free_indices: Any, active_counts: Any, free_counts: Any, n_boxes: int, n_particles: int, request_capacity: int, device: Any) -> tuple[Any, Any, Any, Any]
```

Validate and apply activation requests for nonempty slot capacity.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L468-L582)

## particula.gpu.kernels.slot_management._activate_requests

*function*

```python
def _activate_requests(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), request_masses: wp.array3d(dtype=wp.float64), request_concentration: wp.array2d(dtype=wp.float64), request_charge: wp.array2d(dtype=wp.float64), requested_counts: wp.array(dtype=wp.int32), free_workspace: wp.array2d(dtype=wp.int32), activated_counts: wp.array(dtype=wp.int32)) -> None
```

Copy validated request prefixes to their precomputed free slots.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L186-L208)

## particula.gpu.kernels.slot_management._build_free_workspace

*function*

```python
def _build_free_workspace(categories: wp.array2d(dtype=wp.int32), workspace: wp.array2d(dtype=wp.int32), free_counts: wp.array(dtype=wp.int32)) -> None
```

Compact ascending free slots into private fixed-shape workspace.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L133-L146)

## particula.gpu.kernels.slot_management._classify_slots

*function*

```python
def _classify_slots(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), categories: wp.array2d(dtype=wp.int32), invalid: wp.array(dtype=wp.int32)) -> None
```

Classify one slot and atomically flag an invalid slot state.

The kernel writes a category for every slot but does not write any
caller-provided diagnostic sidecar.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L43-L91)

## particula.gpu.kernels.slot_management._get_required_field

*function*

```python
def _get_required_field(particles: Any, name: str) -> Any
```

Return a required classification field or raise a stable error.

Field access is isolated so malformed particle containers consistently
produce ``ValueError`` rather than leaking attribute-access exceptions.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L237-L246)

## particula.gpu.kernels.slot_management._validate_activation_inputs

*function*

```python
def _validate_activation_inputs(particles: Any, request_masses: Any, request_concentration: Any, request_charge: Any, requested_counts: Any, activated_counts: Any, free_indices: Any, active_counts: Any, free_counts: Any) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, int, int, int, Any]
```

Validate schemas and storage ownership for activation inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L320-L431)

## particula.gpu.kernels.slot_management._validate_activation_ownership

*function*

```python
def _validate_activation_ownership(read_arrays: tuple[Any, ...], writable_arrays: tuple[Any, ...]) -> None
```

Reject direct and partial overlap with mutable activation targets.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L615-L641)

## particula.gpu.kernels.slot_management._validate_activation_preflight

*function*

```python
def _validate_activation_preflight(categories: wp.array2d(dtype=wp.int32), free_counts: wp.array(dtype=wp.int32), request_masses: wp.array3d(dtype=wp.float64), request_concentration: wp.array2d(dtype=wp.float64), request_charge: wp.array2d(dtype=wp.float64), requested_counts: wp.array(dtype=wp.int32), invalid_count: wp.array(dtype=wp.int32), insufficient_capacity: wp.array(dtype=wp.int32), invalid_request: wp.array(dtype=wp.int32)) -> None
```

Validate state, counts, capacity, and selected request records.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L149-L183)

## particula.gpu.kernels.slot_management._validate_array_schema

*function*

```python
def _validate_array_schema(values: Any, name: str, dtype: Any, ndim: int, shape: tuple[int, ...] | None = None, device: Any | None = None) -> Any
```

Validate one Warp particle-field or diagnostic-sidecar schema.

This performs metadata validation only. It neither scans field values nor
mutates the supplied array.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L249-L279)

## particula.gpu.kernels.slot_management._validate_empty_activation_extent

*function*

```python
def _validate_empty_activation_extent(requested_counts: wp.array(dtype=wp.int32), invalid_count: wp.array(dtype=wp.int32), insufficient_capacity: wp.array(dtype=wp.int32), request_capacity: wp.int32) -> None
```

Reject nonzero counts when no request or destination extent exists.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L221-L234)

## particula.gpu.kernels.slot_management._validate_particles

*function*

```python
def _validate_particles(particles: Any) -> tuple[Any, Any, Any, int, int, Any]
```

Validate classification fields and derive the mass-authoritative schema.

Only masses, concentration, and charge participate in classification;
density and volume remain unobserved.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L282-L317)

## particula.gpu.kernels.slot_management._validate_zero_capacity_activation

*function*

```python
def _validate_zero_capacity_activation(requested_counts: Any, request_capacity: int, active_counts: Any, free_counts: Any, device: Any) -> tuple[Any, Any]
```

Validate count and capacity constraints when no particle slots exist.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L434-L465)

## particula.gpu.kernels.slot_management._warp_array_memory_range

*function*

```python
def _warp_array_memory_range(array: Any) -> tuple[int, int] | None
```

Return a supported contiguous Warp array's byte range.

Views are rejected because a byte range cannot distinguish their gaps from
owned storage. Metadata validation has already established the required
dtype before this helper is called.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L585-L612)

## particula.gpu.kernels.slot_management._write_activated_counts

*function*

```python
def _write_activated_counts(requested_counts: wp.array(dtype=wp.int32), activated_counts: wp.array(dtype=wp.int32)) -> None
```

Copy validated requested counts into caller-owned output storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L211-L218)

## particula.gpu.kernels.slot_management._write_diagnostics

*function*

```python
def _write_diagnostics(categories: wp.array2d(dtype=wp.int32), free_indices: wp.array2d(dtype=wp.int32), active_counts: wp.array(dtype=wp.int32), free_counts: wp.array(dtype=wp.int32)) -> None
```

Write per-box counts and ascending free indices from slot categories.

Unused free-index entries are overwritten with ``-1``.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L94-L119)

## particula.gpu.kernels.slot_management._write_empty_diagnostics

*function*

```python
def _write_empty_diagnostics(active_counts: wp.array(dtype=wp.int32), free_counts: wp.array(dtype=wp.int32)) -> None
```

Set both diagnostic counts to zero for a zero-capacity box.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L122-L130)

## particula.gpu.kernels.slot_management.activate_slots_gpu

*function*

```python
def activate_slots_gpu(particles: Any, request_masses: Any, request_concentration: Any, request_charge: Any, requested_counts: Any, activated_counts: Any, free_indices: Any, active_counts: Any, free_counts: Any) -> tuple[Any, Any, Any, Any]
```

Copy selected Warp request prefixes into ascending free slots.

For each box, request rank ``r`` is copied into the ``r``-th ascending
free fixed-capacity slot. Complete preflight validates metadata, storage
ownership, existing slot state, counts, capacity, and only selected
request-prefix records before the mutation writer launches. Valid calls
write the supplied sidecars; ``activated_counts`` equals
``requested_counts``, and diagnostics describe the post-activation state.

This direct boundary never implicitly transfers caller-owned simulation
arrays and accesses only particle masses, concentration, and charge.
Validation may synchronize and read back private scalar status buffers.
Rejected calls leave accessible caller-owned inputs and outputs unchanged.
Rollback is not guaranteed after the mutation writer launches.

**Parameters**

- `particles` (`Any`) — Same-device fixed-capacity storage with float64 ``masses`` shaped ``(n_boxes, n_particles, n_species)`` and ``concentration`` and ``charge`` shaped ``(n_boxes, n_particles)``.
- `request_masses` (`Any`) — Same-device float64 request masses shaped ``(n_boxes, request_capacity, n_species)``.
- `request_concentration` (`Any`) — Same-device float64 request concentrations shaped ``(n_boxes, request_capacity)``.
- `request_charge` (`Any`) — Same-device float64 request charges shaped ``(n_boxes, request_capacity)``.
- `requested_counts` (`Any`) — Same-device int32 selected-prefix lengths shaped ``(n_boxes,)``.
- `activated_counts` (`Any`) — Caller-owned same-device int32 output shaped ``(n_boxes,)``.
- `free_indices` (`Any`) — Caller-owned same-device int32 output shaped ``(n_boxes, n_particles)``; free slots are ascending and its unused entries are ``-1``.
- `active_counts` (`Any`) — Caller-owned same-device int32 output shaped ``(n_boxes,)``.
- `free_counts` (`Any`) — Caller-owned same-device int32 output shaped ``(n_boxes,)``.

**Returns**

- (`Any`) — The exact supplied ``(activated_counts, free_indices, active_counts,
- (`Any`) — free_counts)`` sidecars.

**Raises**

- `ValueError` — If schemas, device or storage ownership, slot state,
selected request records, counts, or capacity are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L747-L865)

## particula.gpu.kernels.slot_management.get_slot_diagnostics_gpu

*function*

```python
def get_slot_diagnostics_gpu(particles: Any, free_indices: Any, active_counts: Any, free_counts: Any) -> tuple[Any, Any, Any]
```

Write read-only slot diagnostics into supplied Warp ``int32`` sidecars.

Import this concrete direct-Warp primitive from
``particula.gpu.kernels.slot_management``; it is not package-exported. A
slot is active iff concentration is finite and positive, all mass lanes
are finite and nonnegative, float64 total mass is finite and positive, and
charge is finite. A slot is free iff concentration, all masses, and charge
are exactly zero. Any other state raises ``ValueError`` before diagnostic
writes. The primitive reads only those classification fields, leaving all
particle storage, including density and volume, unchanged.

**Parameters**

- `particles` (`Any`) — Container with same-device float64 ``masses`` shaped ``(n_boxes, n_particles, n_species)``, plus concentration and charge fields shaped ``(n_boxes, n_particles)``.
- `free_indices` (`Any`) — Same-device int32 output shaped ``(n_boxes, n_particles)``.
- `active_counts` (`Any`) — Same-device int32 output shaped ``(n_boxes,)``.
- `free_counts` (`Any`) — Same-device int32 output shaped ``(n_boxes,)``.

**Returns**

- (`tuple[Any, Any, Any]`) — The exact supplied ``(free_indices, active_counts, free_counts)``.

**Raises**

- `ValueError` — If a schema is malformed or any slot is neither active nor
free.

> **Notes**
>
> Valid calls overwrite every diagnostic entry. Free indices are
> ascending and unused positions are ``-1``. Schema and state rejection
> occurs before caller-owned diagnostics are written. Callers own device
> placement and synchronization before reading device-resident diagnostics
> on the host.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/slot_management.py#L644-L744)
