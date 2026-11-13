# particula.gpu.conversion

*module*

Convert simulation data containers between CPU and Warp representations.

This module provides helpers to transfer CPU-side particle, gas, and
environment containers to Warp arrays and reconstruct their CPU-side
equivalents. The conversion functions enable manual control over device
selection, copy behavior, and synchronization for long GPU-resident
simulations.

The ``WarpGasData.partitioning`` mirror is a binary per-box int32 mask. CPU
conversion expands the shared ``GasData`` species mask to every box; restoring
requires those box masks to remain identical before collapsing them to the CPU
representation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/conversion.py#L1-L1)

## particula.gpu.conversion._ensure_warp_available

*function*

```python
def _ensure_warp_available()
```

Ensure Warp is available, raise RuntimeError if not.

**Returns**

- `result` (`ModuleType`) — The warp module if available.

**Raises**

- `RuntimeError` — If Warp is not installed.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/conversion.py#L35-L51)

## particula.gpu.conversion._from_numpy_zero_copy

*function*

```python
def _from_numpy_zero_copy(wp, values, dtype, device: str)
```

Call ``wp.from_numpy`` with an explicit zero-copy preference.

Some Warp versions accept ``copy=False`` explicitly, while older versions
reject the keyword and use their built-in default behavior. Prefer the
explicit call when supported so regression tests can guard the intended
branch, but fall back for compatibility with older Warp releases.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/conversion.py#L82-L95)

## particula.gpu.conversion._restore_partitioning_bool

*function*

```python
def _restore_partitioning_bool(partitioning_values: NDArray[np.int32] | NDArray[np.float64]) -> NDArray[np.bool_]
```

Validate GPU partitioning flags before restoring CPU bool values.

**Parameters**

- `partitioning_values` (`NDArray[np.int32] | NDArray[np.float64]`) — GPU-restored partitioning values.

**Returns**

- (`NDArray[np.bool_]`) — Partitioning values restored as a NumPy bool array.

**Raises**

- `ValueError` — If any value is not one of ``0`` or ``1``.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/conversion.py#L98-L117)

## particula.gpu.conversion._validate_device

*function*

```python
def _validate_device(wp, device: str)
```

Validate device string and return device object.

**Parameters**

- `wp` (`ModuleType`) — The warp module.
- `device` (`str`) — Device string (e.g., "cuda", "cuda:0", "cpu").

**Returns**

- `result` (`warp.context.Device`) — Warp device object.

**Raises**

- `RuntimeError` — If device is not found.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/conversion.py#L54-L79)

## particula.gpu.conversion.from_warp_environment_data

*function*

```python
def from_warp_environment_data(gpu_data: 'WarpEnvironmentData', sync: bool = True) -> 'EnvironmentData'
```

Transfer WarpEnvironmentData back to CPU.

Use this to transfer GPU-resident environment data back to CPU after
GPU simulation steps. The returned EnvironmentData can be used for
checkpointing, analysis, or continuing with CPU-based operations.

**Parameters**

- `gpu_data` (`'WarpEnvironmentData'`) — GPU-resident WarpEnvironmentData container.
- `sync` (`bool`) (default: `True`) — If True (default), synchronize device before transfer to ensure all GPU operations have completed. Set False only if you've already synchronized manually (e.g., for batched transfers).

**Returns**

- (`'EnvironmentData'`) — CPU-side EnvironmentData with NumPy arrays.

**Raises**

- `ValueError` — If the recovered arrays do not satisfy the
``EnvironmentData`` CPU schema.

> **Example**
>
> >>> result = from_warp_environment_data(gpu_environment)
> >>>
> >>> import warp as wp
> >>> wp.synchronize()
> >>> result = from_warp_environment_data(gpu_environment, sync=False)

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/conversion.py#L611-L653)

## particula.gpu.conversion.from_warp_gas_data

*function*

```python
def from_warp_gas_data(gpu_data: 'WarpGasData', name: list[str] | None = None, sync: bool = True, *, allow_per_box_partitioning: bool = False) -> 'GasData'
```

Transfer WarpGasData back to CPU with an intentionally lossy restore.

Use this to transfer GPU-resident gas data back to CPU after
GPU simulation steps. The returned GasData can be used for
checkpointing, analysis, or continuing with CPU-based operations.

> **Note**
>
> The restored CPU ``GasData`` includes the current GPU concentration,
> including in-place condensation coupling. It omits ``vapor_pressure``
> because that field is not part of the CPU schema. The ``WarpGasData``
> container may
> still retain ``gpu_data.vapor_pressure`` after this helper returns, so
> callers can keep reading that GPU-side field while they retain the GPU
> container, or preserve a sidecar copy if they need a detached CPU copy.
> ``from_warp_gas_data()`` restores only the CPU-owned ``GasData``
> fields.
> 
> ``WarpGasData`` does not store species names because string data is
> not GPU-compatible. Prefer caller-supplied ordered names when
> reconstructing ``GasData``. Placeholder names are generated only
> when ``name`` is omitted or explicitly set to ``None``.
> 
> Restored ``partitioning`` must be a binary ``(n_boxes, n_species)``
> mask that is identical in every box because CPU ``GasData`` owns one
> shared species mask. Any incompatible values or layout raise
> ``ValueError`` before a ``GasData`` instance is returned.

**Parameters**

- `gpu_data` (`'WarpGasData'`) — GPU-resident WarpGasData container.
- `name` (`list[str] | None`) (default: `None`) — Optional ordered species names supplied by the caller. If omitted or ``None``, generates placeholder names such as ``["species_0", "species_1", ...]``. If provided, the list length must match ``gpu_data.molar_mass.shape[0]``.
- `sync` (`bool`) (default: `True`) — If True (default), synchronize device before transfer to ensure all GPU operations have completed. Set False only if you've already synchronized manually.
- `allow_per_box_partitioning` (`bool`) (default: `False`) — Internal checkpoint-only escape hatch for building a lossy CPU inspection carrier from a nonuniform GPU mask.

**Returns**

- (`'GasData'`) — CPU-side GasData with NumPy arrays.

**Raises**

- `ValueError` — If ``name`` is not ``None`` or a ``list[str]``.
- `ValueError` — If the supplied name count does not match ``n_species``.
- `ValueError` — If restored ``partitioning`` is non-binary, is not a
two-dimensional ``(n_boxes, n_species)`` mask, or differs by box.

> **Example**
>
> >>> # With caller-supplied ordered names
> >>> result = from_warp_gas_data(gpu_data, name=["Water", "H2SO4"])
> >>>
> >>> # With placeholder names because GPU data stores no names
> >>> result = from_warp_gas_data(gpu_data)
> >>> print(result.name)  # ["species_0", "species_1"]

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/conversion.py#L489-L608)

## particula.gpu.conversion.from_warp_particle_data

*function*

```python
def from_warp_particle_data(gpu_data: 'WarpParticleData', sync: bool = True) -> 'ParticleData'
```

Restore WarpParticleData with its supported population provenance.

Use this to transfer GPU-resident particle data back to CPU after
GPU simulation steps. The returned ParticleData can be used for
analysis or continuing with CPU-based operations. Discrete and resolved
tags survive restoration without density conversion. Missing, untagged,
unknown tags and unsupported PDF metadata reject before
synchronization or readback.

**Parameters**

- `gpu_data` (`'WarpParticleData'`) — GPU-resident WarpParticleData container.
- `sync` (`bool`) (default: `True`) — If True (default), synchronize device before transfer to ensure all GPU operations have completed. Set False only if you've already synchronized manually (e.g., for batched transfers).

**Returns**

- (`'ParticleData'`) — CPU-side ParticleData with NumPy arrays and preserved population kind.

> **Example**
>
> >>> # After GPU simulation
> >>> result = from_warp_particle_data(gpu_data)
> >>>
> >>> # Batched transfers with manual sync (advanced)
> >>> import warp as wp
> >>> wp.synchronize()
> >>> data1 = from_warp_particle_data(gpu_data1, sync=False)
> >>> data2 = from_warp_particle_data(gpu_data2, sync=False)

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/conversion.py#L436-L486)

## particula.gpu.conversion.gpu_context

*function*

```python
def gpu_context(data: 'ParticleData', device: str = 'cuda') -> Generator['WarpParticleData', None, None]
```

Context manager for scoped GPU-resident simulation.

Transfers ParticleData to GPU on entry. User is responsible for
calling from_warp_particle_data() when ready to transfer back
(typically inside the context or after exit).

This is a convenience wrapper for simple GPU simulation patterns.
For complex workflows with multiple data containers or explicit
sync control, use to_warp_particle_data()/from_warp_particle_data()
directly.

**Parameters**

- `data` (`'ParticleData'`) — CPU-side ParticleData to transfer to GPU.
- `device` (`str`) (default: `'cuda'`) — Target GPU device ("cuda", "cuda:0", "cpu").

**Yields**

- (`'WarpParticleData'`) — WarpParticleData on the specified device.

> **Example**
>
> >>> # Transfer back inside context
> >>> with gpu_context(particles) as gpu_data:
> ...     for _ in range(1000):
> ...         gpu_data = physics_step(gpu_data, dt)
> ...     result = from_warp_particle_data(gpu_data)
> >>>
> >>> # Or transfer back after context (data still valid)
> >>> with gpu_context(particles) as gpu_data:
> ...     for _ in range(1000):
> ...         gpu_data = physics_step(gpu_data, dt)
> >>> result = from_warp_particle_data(gpu_data)

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/conversion.py#L656-L693)

## particula.gpu.conversion.to_per_box_partitioning

*function*

```python
def to_per_box_partitioning(partitioning: np.ndarray, n_boxes: int, *, device: str = 'cpu') -> Any
```

Migrate a legacy shared mask to the Warp per-box mask schema.

``WarpGasData.partitioning`` is version-2 public state: its active-device
representation is always binary ``int32`` with shape
``(n_boxes, n_species)``.  This explicit helper is the compatibility path
for callers that previously kept a one-dimensional shared mask.

**Parameters**

- `partitioning` (`np.ndarray`) — Legacy ``(n_species,)`` or current ``(n_boxes, n_species)`` binary mask.
- `n_boxes` (`int`) — Number of simulation boxes.
- `device` (`str`) (default: `'cpu'`) — Warp device for the returned mask.

**Returns**

- (`Any`) — Active-device binary ``int32`` mask with per-box shape.

**Raises**

- `ValueError` — If shape or binary values are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/conversion.py#L394-L433)

## particula.gpu.conversion.to_warp_environment_data

*function*

```python
def to_warp_environment_data(data: 'EnvironmentData', device: str = 'cuda', copy: bool = True) -> 'WarpEnvironmentData'
```

Transfer EnvironmentData to GPU with explicit control.

Use this helper when environment state should remain on the selected
Warp device alongside particle or gas data during GPU-resident
simulations.

**Parameters**

- `data` (`'EnvironmentData'`) — CPU-side EnvironmentData container.
- `device` (`str`) (default: `'cuda'`) — Target device ("cuda", "cuda:0", "cuda:1", "cpu").
- `copy` (`bool`) (default: `True`) — If True (default), always copy data to device. If False, attempt zero-copy via wp.from_numpy() when arrays are already on a compatible device.

**Returns**

- (`'WarpEnvironmentData'`) — WarpEnvironmentData with Warp arrays on specified device.

**Raises**

- `RuntimeError` — If Warp is not available or device not found.

> **Example**
>
> >>> from particula.gpu import to_warp_environment_data
> >>> gpu_environment = to_warp_environment_data(
> ...     environment_data, device="cpu"
> ... )
> >>> gpu_environment.temperature.shape
> (1,)

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/conversion.py#L202-L262)

## particula.gpu.conversion.to_warp_gas_data

*function*

```python
def to_warp_gas_data(data: 'GasData', device: str = 'cuda', copy: bool = True, vapor_pressure: NDArray[np.float64] | None = None) -> 'WarpGasData'
```

Transfer GasData to GPU with caller-owned vapor-pressure state.

Use this for long GPU-resident simulations where you want to:
1. Transfer data to GPU once at simulation start
2. Run many timesteps without CPU round-trips
3. Transfer back only when needed (checkpoints, final result)

> **Note**
>
> The 'name' field from GasData is excluded (strings are not
> GPU-compatible). Use index mapping for species identification.
> 
> The shared species-level CPU ``partitioning`` field is expanded to an
> identical per-box ``(n_boxes, n_species)`` mask and converted from bool
> to binary int32 (1 = True, 0 = False) for GPU compatibility.
> 
> ``WarpGasData`` requires a ``vapor_pressure`` buffer even though CPU
> ``GasData`` does not own that field. Callers may supply an initial
> ``(n_boxes, n_species)`` vapor-pressure array for kernels that consume
> it directly. ``condensation_step_gpu()`` instead refreshes this buffer
> from its thermodynamic configuration on every substep. That step also
> updates this mirror's ``concentration`` from each inventory-limited
> particle
> transfer. The updated concentration informs the next mass-transfer
> proposal, not vapor-pressure refresh.
> 
> If ``vapor_pressure`` is omitted, this helper allocates a zero-filled
> ``(n_boxes, n_species)`` GPU buffer on the selected device. That
> default keeps the transfer schema valid. It is suitable for
> ``condensation_step_gpu()``, which overwrites vapor pressure before
> calculating transfer and couples finalized transfer to gas
> concentration, but not for kernels that consume the field without
> refreshing it.

**Parameters**

- `data` (`'GasData'`) — CPU-side GasData container.
- `device` (`str`) (default: `'cuda'`) — Target device ("cuda", "cuda:0", "cuda:1", "cpu").
- `copy` (`bool`) (default: `True`) — If True (default), always copy data to device.   If False, attempt zero-copy via wp.from_numpy() when   arrays are already on a compatible device.
- `vapor_pressure` (`NDArray[np.float64] | None`) (default: `None`) — Optional caller-supplied vapor pressure array in Pa. Shape: ``(n_boxes, n_species)``. If omitted, this helper creates a zero-filled GPU buffer with that same shape for the Warp mirror.

**Returns**

- (`'WarpGasData'`) — WarpGasData with Warp arrays on specified device.

**Raises**

- `RuntimeError` — If Warp is not available or device not found.
- `ValueError` — If vapor_pressure shape doesn't match expected
(n_boxes, n_species).
- `ValueError` — If vapor_pressure contains NaN, inf, or negative values.

> **Example**
>
> >>> from particula.gpu import to_warp_gas_data
> >>> # Omitted vapor pressure allocates zeros for schema parity only.
> >>> gpu_gas = to_warp_gas_data(gas_data, device="cuda")
> >>> # Supply initial vapor pressure for kernels that do not refresh it:
> >>> vp = np.array([[1000.0, 500.0, 200.0], [1000.0, 500.0, 200.0]])
> >>> gpu_gas = to_warp_gas_data(gas_data, vapor_pressure=vp)

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/conversion.py#L265-L391)

## particula.gpu.conversion.to_warp_particle_data

*function*

```python
def to_warp_particle_data(data: 'ParticleData', device: str = 'cuda', copy: bool = True) -> 'WarpParticleData'
```

Transfer ParticleData to Warp without changing population values.

Use this for long GPU-resident simulations where you want to:
1. Transfer data to GPU once at simulation start
2. Run many timesteps without CPU round-trips
3. Transfer back only when needed (checkpoints, final result)

Explicit discrete and resolved distributions retain their kind. Tagged
PDFs reject before Warp availability checks or uploads because no GPU
radius-grid algorithm is supported. Untagged storage is not transferable.

**Parameters**

- `data` (`'ParticleData'`) — CPU-side ParticleData container.
- `device` (`str`) (default: `'cuda'`) — Target device ("cuda", "cuda:0", "cuda:1", "cpu").
- `copy` (`bool`) (default: `True`) — If True (default), always copy data to device.   If False, attempt zero-copy via wp.from_numpy() when   arrays are already on a compatible device.

**Returns**

- (`'WarpParticleData'`) — WarpParticleData with Warp arrays on specified device.

**Raises**

- `ValueError` — If tagged data is malformed or uses continuous_pdf.
- `RuntimeError` — If Warp is not available or device not found.

> **Example**
>
> >>> from particula.gpu import to_warp_particle_data
> >>> gpu_data = to_warp_particle_data(particles, device="cuda")
> >>> len(gpu_data.masses.shape)
> 3

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/conversion.py#L120-L199)
