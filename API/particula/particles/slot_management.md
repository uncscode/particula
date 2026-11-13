# particula.particles.slot_management

*module*

Discover and activate fixed particle slots in CPU particle data.

This module provides the CPU reference classification for active and free
particle-resolved slots. Its discovery API returns fixed-shape, newly
allocated diagnostics without changing particle storage. Its activation API
maps validated request prefixes into the ascending free slots after complete
read-only preflight.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/slot_management.py#L1-L1)

## particula.particles.slot_management._validate_destinations

*function*

```python
def _validate_destinations(data: ParticleData) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
```

Validate and return writable activation destination arrays.

**Parameters**

- `data` (`ParticleData`) — Particle data whose mutable mass, concentration, and charge storage is validated.

**Returns**

- (`tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]`) — The validated mass, concentration, and charge arrays, in that order.

**Raises**

- `ValueError` — If a mutable destination is not a writable ``float64``
NumPy array, storage overlaps another mutable or protected field,
or the destination shapes are incompatible.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/slot_management.py#L89-L139)

## particula.particles.slot_management._validate_preflight

*function*

```python
def _validate_preflight(request_fields: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], destination_fields: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], requested_counts: NDArray[np.integer], free_counts: NDArray[np.int32]) -> None
```

Validate storage isolation, capacity, and selected request records.

Only request records within each box's declared prefix are inspected.
This validation performs no writes so that activation remains atomic when
a later box or record is invalid.

**Parameters**

- `request_fields` (`tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]`) — Validated request mass, concentration, and charge arrays.
- `destination_fields` (`tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]`) — Validated mutable mass, concentration, and charge destination arrays.
- `requested_counts` (`NDArray[np.integer]`) — Per-box request prefix lengths.
- `free_counts` (`NDArray[np.int32]`) — Number of currently free slots in each box.

**Raises**

- `ValueError` — If request storage aliases a destination, a requested
prefix exceeds free capacity, or a selected record is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/slot_management.py#L202-L255)

## particula.particles.slot_management._validate_requests

*function*

```python
def _validate_requests(request_masses: NDArray[np.float64], request_concentration: NDArray[np.float64], request_charge: NDArray[np.float64], requested_counts: NDArray[np.integer], n_boxes: int, n_species: int) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
```

Validate request schemas and return their fields.

**Parameters**

- `request_masses` (`NDArray[np.float64]`) — Requested per-species mass records.
- `request_concentration` (`NDArray[np.float64]`) — Requested concentration records.
- `request_charge` (`NDArray[np.float64]`) — Requested charge records.
- `requested_counts` (`NDArray[np.integer]`) — Per-box prefix lengths to activate.
- `n_boxes` (`int`) — Required number of request boxes.
- `n_species` (`int`) — Required number of mass lanes in each request record.

**Returns**

- (`NDArray[np.float64]`) — The validated request mass, concentration, and charge arrays, in that
- (`NDArray[np.float64]`) — order.

**Raises**

- `ValueError` — If request fields are not shaped ``float64`` NumPy arrays,
counts are not a valid integer vector, or a count exceeds request
capacity.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/slot_management.py#L142-L199)

## particula.particles.slot_management.activate_slots

*function*

```python
def activate_slots(data: ParticleData, request_masses: NDArray[np.float64], request_concentration: NDArray[np.float64], request_charge: NDArray[np.float64], requested_counts: NDArray[np.integer]) -> NDArray[np.int32]
```

Activate ascending free slots from per-box request prefixes.

Request rank ``r`` in each box is copied into that box's ``r``-th
ascending free slot. Only records below the corresponding requested count
are validated or read. The destination fields mutate in place after a
complete global preflight, and a newly allocated ``np.int32`` count array
is returned.

**Parameters**

- `data` (`ParticleData`) — Particle storage with writable float64 mass, concentration, and charge arrays.
- `request_masses` (`NDArray[np.float64]`) — Requested per-species masses with shape ``(n_boxes, request_capacity, n_species)``.
- `request_concentration` (`NDArray[np.float64]`) — Requested concentrations with shape ``(n_boxes, request_capacity)``.
- `request_charge` (`NDArray[np.float64]`) — Requested charges with shape ``(n_boxes, request_capacity)``.
- `requested_counts` (`NDArray[np.integer]`) — Integer prefix counts with shape ``(n_boxes,)``.

**Returns**

- (`NDArray[np.int32]`) — A fresh ``np.int32`` array containing the activated count per box.

**Raises**

- `ValueError` — If an input schema, destination, slot state, requested
prefix, or available free capacity is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/slot_management.py#L258-L325)

## particula.particles.slot_management.get_slot_diagnostics

*function*

```python
def get_slot_diagnostics(data: ParticleData) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]
```

Return read-only fixed-shape diagnostics for particle slots.

A slot is active when its concentration is positive and finite, each mass
lane is finite and nonnegative, and its total mass is positive and finite.
A slot is free when its concentration, all mass lanes, and its charge are
exactly zero. All other slot states are invalid. This function does not
modify ``data`` and allocates fresh output arrays for every call.

**Parameters**

- `data` (`ParticleData`) — Fixed-shape particle data to inspect without mutation.

**Returns**

- (`NDArray[np.int32]`) — A tuple ``(free_indices, active_counts, free_counts)`` of newly
- (`NDArray[np.int32]`) — allocated ``np.int32`` arrays. ``free_indices`` has shape
- (`NDArray[np.int32]`) — ``(n_boxes, n_particles)``; each row contains ascending free-slot
- (`tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]`) — indices followed by ``-1``. The count arrays have shape ``(n_boxes,)``.

**Raises**

- `ValueError` — If any particle slot is neither active nor free. The error
message is ``"Invalid particle slot state."``.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/slot_management.py#L18-L86)
