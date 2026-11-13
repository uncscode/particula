# particula.gpu.kernels.exhaustion

*module*

Provide fixed-capacity direct Warp exhaustion helpers.

This concrete-module-only boundary consumes already-resolved per-box release
counts. It has no policy resolution, CPU fallback, host particle transfer,
resizing, or runnable wrapper. Callers own every particle-scale planning and
diagnostic buffer. Read-only preflight precedes planning; planning can replace
documented buffer contents, and one commit launches only after every box passes
its diagnostics.

The planner uses a deterministic bitonic sorting network and a linear
equal-weight interval sweep. Sorting costs ``O(B * N * log2(N)**2)``
compare-exchanges; all other planning and commit work costs ``O(B * N * S)``.
The caller supplies its entire particle-scale scratch footprint through
:class:`ResamplingBuffers`.

The concrete-only volume scaling representative-volume helper validates all
caller-owned
State before writing its diagnostic or scaling selected rows. It writes the
resolved requested factor for selected rows and ``1.0`` otherwise. It has no
package export, policy, transfer, resizing, or runnable integration; callers
own device placement and explicitly synchronize before observing results.

Direct and prepared entry points require explicit particle-resolved metadata.
Untagged, discrete, and continuous-PDF populations reject before payload access.
Prepared records pin native validation and workspace allocation in setup.
Enqueue dispatches retained device work only, without repeated host
metadata validation, readback, allocation, transfer, synchronization, lookup,
rebinding, or recovery. Its established device-side dynamic validation runs
against retained references. Rejection is pre-writer; rollback is not promised
after a writer launches.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1-L1)

## particula.gpu.kernels.exhaustion.PLANNING_DIVERSITY_FAILURE

*attribute* · *module attribute*

```python
PLANNING_DIVERSITY_FAILURE = 5
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L61-L61)

## particula.gpu.kernels.exhaustion.PLANNING_EXACT_INVENTORY_FAILURE

*attribute* · *module attribute*

```python
PLANNING_EXACT_INVENTORY_FAILURE = 1
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L57-L57)

## particula.gpu.kernels.exhaustion.PLANNING_MEAN_RADIUS_FAILURE

*attribute* · *module attribute*

```python
PLANNING_MEAN_RADIUS_FAILURE = 3
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L59-L59)

## particula.gpu.kernels.exhaustion.PLANNING_NONFINITE_FAILURE

*attribute* · *module attribute*

```python
PLANNING_NONFINITE_FAILURE = 6
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L62-L62)

## particula.gpu.kernels.exhaustion.PLANNING_RADIUS_CUBED_FAILURE

*attribute* · *module attribute*

```python
PLANNING_RADIUS_CUBED_FAILURE = 2
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L58-L58)

## particula.gpu.kernels.exhaustion.PLANNING_SUCCESS

*attribute* · *module attribute*

```python
PLANNING_SUCCESS = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L56-L56)

## particula.gpu.kernels.exhaustion.PLANNING_SURFACE_FAILURE

*attribute* · *module attribute*

```python
PLANNING_SURFACE_FAILURE = 4
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L60-L60)

## particula.gpu.kernels.exhaustion._PI

*attribute* · *module attribute*

```python
_PI = wp.constant(wp.float64(np.pi))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L64-L64)

## particula.gpu.kernels.exhaustion.ResamplingBuffers

*class* · *dataclass*

```python
class ResamplingBuffers
```

Hold caller-owned fixed-shape Warp storage for resampling.

The record is concrete-module-only and does not bind a reusable CPU plan
to particle state. Every field must be a distinct, same-device Warp array
with the shape implied by particle boxes ``B``, capacity ``N``, and species
count ``S``. Successful nonzero-demand planning initializes every output
and scratch lane deterministically; zero-demand buffer lanes are unchanged.

**Attributes**

- `retained_counts` (`Any`) — ``int32`` array shaped ``(B,)`` of planned retained active slots.
- `released_counts` (`Any`) — ``int32`` array shaped ``(B,)`` of planned releases.
- `retained_indices` (`Any`) — ``int32`` array shaped ``(B, N)`` of retained original slot indices, with unused lanes set to ``-1``.
- `released_indices` (`Any`) — ``int32`` array shaped ``(B, N)`` of released original slot indices, with unused lanes set to ``-1``.
- `sorted_indices` (`Any`) — ``int32`` array shaped ``(B, N)`` used for the sorted source sequence, with unused lanes set to ``-1``.
- `replacement_masses` (`Any`) — ``float64`` array shaped ``(B, N, S)`` containing planned per-particle species masses; unused rows are zero.
- `replacement_concentration` (`Any`) — ``float64`` array shaped ``(B, N)`` of planned concentrations; unused lanes are zero.
- `replacement_charge` (`Any`) — ``float64`` array shaped ``(B, N)`` of planned charges; unused lanes are zero.
- `source_radii` (`Any`) — ``float64`` array shaped ``(B, N)`` for planning radii.
- `radius_cubed_relative_error` (`Any`) — ``float64`` array shaped ``(B,)`` of radius-cubed diagnostic errors.
- `mean_radius_relative_error` (`Any`) — ``float64`` array shaped ``(B,)`` of mean-radius diagnostic errors.
- `surface_relative_error` (`Any`) — ``float64`` array shaped ``(B,)`` of surface area diagnostic errors.
- `diversity_absolute_error` (`Any`) — ``float64`` array shaped ``(B,)`` of Riemer-diversity diagnostic errors.
- `planning_status` (`Any`) — ``int32`` array shaped ``(B,)`` with ``0`` for success, ``1`` for inventory failure, ``2`` through ``5`` for radius-cubed, mean-radius, surface, and diversity failures, and ``6`` for non-finite planning results.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L67-L121)

### particula.gpu.kernels.exhaustion.ResamplingBuffers.diversity_absolute_error

*attribute* · *instance attribute*

```python
diversity_absolute_error: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L120-L120)

### particula.gpu.kernels.exhaustion.ResamplingBuffers.mean_radius_relative_error

*attribute* · *instance attribute*

```python
mean_radius_relative_error: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L118-L118)

### particula.gpu.kernels.exhaustion.ResamplingBuffers.planning_status

*attribute* · *instance attribute*

```python
planning_status: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L121-L121)

### particula.gpu.kernels.exhaustion.ResamplingBuffers.radius_cubed_relative_error

*attribute* · *instance attribute*

```python
radius_cubed_relative_error: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L117-L117)

### particula.gpu.kernels.exhaustion.ResamplingBuffers.released_counts

*attribute* · *instance attribute*

```python
released_counts: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L109-L109)

### particula.gpu.kernels.exhaustion.ResamplingBuffers.released_indices

*attribute* · *instance attribute*

```python
released_indices: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L111-L111)

### particula.gpu.kernels.exhaustion.ResamplingBuffers.replacement_charge

*attribute* · *instance attribute*

```python
replacement_charge: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L115-L115)

### particula.gpu.kernels.exhaustion.ResamplingBuffers.replacement_concentration

*attribute* · *instance attribute*

```python
replacement_concentration: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L114-L114)

### particula.gpu.kernels.exhaustion.ResamplingBuffers.replacement_masses

*attribute* · *instance attribute*

```python
replacement_masses: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L113-L113)

### particula.gpu.kernels.exhaustion.ResamplingBuffers.retained_counts

*attribute* · *instance attribute*

```python
retained_counts: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L108-L108)

### particula.gpu.kernels.exhaustion.ResamplingBuffers.retained_indices

*attribute* · *instance attribute*

```python
retained_indices: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L110-L110)

### particula.gpu.kernels.exhaustion.ResamplingBuffers.sorted_indices

*attribute* · *instance attribute*

```python
sorted_indices: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L112-L112)

### particula.gpu.kernels.exhaustion.ResamplingBuffers.source_radii

*attribute* · *instance attribute*

```python
source_radii: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L116-L116)

### particula.gpu.kernels.exhaustion.ResamplingBuffers.surface_relative_error

*attribute* · *instance attribute*

```python
surface_relative_error: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L119-L119)

### particula.gpu.kernels.exhaustion.ResamplingBuffers.__init__

*method*

```python
def __init__(retained_counts: Any, released_counts: Any, retained_indices: Any, released_indices: Any, sorted_indices: Any, replacement_masses: Any, replacement_concentration: Any, replacement_charge: Any, source_radii: Any, radius_cubed_relative_error: Any, mean_radius_relative_error: Any, surface_relative_error: Any, diversity_absolute_error: Any, planning_status: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L0-L0)

## particula.gpu.kernels.exhaustion._PreparedRepresentativeVolumeScalingCall

*class* · *dataclass*

```python
class _PreparedRepresentativeVolumeScalingCall
```

Pin one allocation-complete representative-scaling sequence.

Setup freezes metadata, arrays, controls, and workspace allocation by
identity. Enqueue uses these retained references for device-side validation,
selection, and scaling only; it performs no host readback, allocation,
transfer, synchronization, lookup, rebinding, or recovery. This private
record is concrete-module-only. Rejection before scaling is pre-writer;
rollback is not promised after a scaling writer launches.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1050-L1078)

### particula.gpu.kernels.exhaustion._PreparedRepresentativeVolumeScalingCall.charge

*attribute* · *instance attribute*

```python
charge: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1065-L1065)

### particula.gpu.kernels.exhaustion._PreparedRepresentativeVolumeScalingCall.concentration

*attribute* · *instance attribute*

```python
concentration: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1064-L1064)

### particula.gpu.kernels.exhaustion._PreparedRepresentativeVolumeScalingCall.density

*attribute* · *instance attribute*

```python
density: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1066-L1066)

### particula.gpu.kernels.exhaustion._PreparedRepresentativeVolumeScalingCall.device

*attribute* · *instance attribute*

```python
device: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1075-L1075)

### particula.gpu.kernels.exhaustion._PreparedRepresentativeVolumeScalingCall.dimensions

*attribute* · *instance attribute*

```python
dimensions: tuple[int, int, int]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1074-L1074)

### particula.gpu.kernels.exhaustion._PreparedRepresentativeVolumeScalingCall.masses

*attribute* · *instance attribute*

```python
masses: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1063-L1063)

### particula.gpu.kernels.exhaustion._PreparedRepresentativeVolumeScalingCall.minimum_scale

*attribute* · *instance attribute*

```python
minimum_scale: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1071-L1071)

### particula.gpu.kernels.exhaustion._PreparedRepresentativeVolumeScalingCall.minimum_volume

*attribute* · *instance attribute*

```python
minimum_volume: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1072-L1072)

### particula.gpu.kernels.exhaustion._PreparedRepresentativeVolumeScalingCall.provisional_source_demand

*attribute* · *instance attribute*

```python
provisional_source_demand: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1068-L1068)

### particula.gpu.kernels.exhaustion._PreparedRepresentativeVolumeScalingCall.requested_scale

*attribute* · *instance attribute*

```python
requested_scale: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1070-L1070)

### particula.gpu.kernels.exhaustion._PreparedRepresentativeVolumeScalingCall.resolved_scale

*attribute* · *instance attribute*

```python
resolved_scale: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1073-L1073)

### particula.gpu.kernels.exhaustion._PreparedRepresentativeVolumeScalingCall.result

*attribute* · *instance attribute*

```python
result: tuple[Any, Any, Any]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1062-L1062)

### particula.gpu.kernels.exhaustion._PreparedRepresentativeVolumeScalingCall.scaling_required

*attribute* · *instance attribute*

```python
scaling_required: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1069-L1069)

### particula.gpu.kernels.exhaustion._PreparedRepresentativeVolumeScalingCall.selected

*attribute* · *instance attribute*

```python
selected: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1077-L1077)

### particula.gpu.kernels.exhaustion._PreparedRepresentativeVolumeScalingCall.status

*attribute* · *instance attribute*

```python
status: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1076-L1076)

### particula.gpu.kernels.exhaustion._PreparedRepresentativeVolumeScalingCall.upstream_invalid

*attribute* · *instance attribute*

```python
upstream_invalid: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1078-L1078)

### particula.gpu.kernels.exhaustion._PreparedRepresentativeVolumeScalingCall.volume

*attribute* · *instance attribute*

```python
volume: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1067-L1067)

### particula.gpu.kernels.exhaustion._PreparedRepresentativeVolumeScalingCall.__init__

*method*

```python
def __init__(result: tuple[Any, Any, Any], masses: Any, concentration: Any, charge: Any, density: Any, volume: Any, provisional_source_demand: Any, scaling_required: Any, requested_scale: Any, minimum_scale: Any, minimum_volume: Any, resolved_scale: Any, dimensions: tuple[int, int, int], device: Any, status: Any, selected: Any, upstream_invalid: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L0-L0)

## particula.gpu.kernels.exhaustion._PreparedResamplingCall

*class* · *dataclass*

```python
class _PreparedResamplingCall
```

Pin one allocation-complete resampling launch sequence.

Setup freezes metadata, arrays, controls, and workspace allocation by
identity. Enqueue uses these retained references for device-side validation,
planning, and commit only; it performs no host readback, allocation,
transfer, synchronization, lookup, rebinding, or recovery. This private
record is concrete-module-only. Rejection before commit is pre-writer;
rollback is not promised after a commit writer launches.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1022-L1047)

### particula.gpu.kernels.exhaustion._PreparedResamplingCall.bounds

*attribute* · *instance attribute*

```python
bounds: tuple[float, float, float, float]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1042-L1042)

### particula.gpu.kernels.exhaustion._PreparedResamplingCall.buffer_arrays

*attribute* · *instance attribute*

```python
buffer_arrays: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1041-L1041)

### particula.gpu.kernels.exhaustion._PreparedResamplingCall.charge

*attribute* · *instance attribute*

```python
charge: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1037-L1037)

### particula.gpu.kernels.exhaustion._PreparedResamplingCall.concentration

*attribute* · *instance attribute*

```python
concentration: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1036-L1036)

### particula.gpu.kernels.exhaustion._PreparedResamplingCall.density

*attribute* · *instance attribute*

```python
density: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1038-L1038)

### particula.gpu.kernels.exhaustion._PreparedResamplingCall.device

*attribute* · *instance attribute*

```python
device: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1044-L1044)

### particula.gpu.kernels.exhaustion._PreparedResamplingCall.dimensions

*attribute* · *instance attribute*

```python
dimensions: tuple[int, int, int]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1043-L1043)

### particula.gpu.kernels.exhaustion._PreparedResamplingCall.invalid

*attribute* · *instance attribute*

```python
invalid: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1045-L1045)

### particula.gpu.kernels.exhaustion._PreparedResamplingCall.masses

*attribute* · *instance attribute*

```python
masses: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1035-L1035)

### particula.gpu.kernels.exhaustion._PreparedResamplingCall.planning_failed

*attribute* · *instance attribute*

```python
planning_failed: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1046-L1046)

### particula.gpu.kernels.exhaustion._PreparedResamplingCall.required_release_counts

*attribute* · *instance attribute*

```python
required_release_counts: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1040-L1040)

### particula.gpu.kernels.exhaustion._PreparedResamplingCall.result

*attribute* · *instance attribute*

```python
result: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1034-L1034)

### particula.gpu.kernels.exhaustion._PreparedResamplingCall.upstream_invalid

*attribute* · *instance attribute*

```python
upstream_invalid: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1047-L1047)

### particula.gpu.kernels.exhaustion._PreparedResamplingCall.volume

*attribute* · *instance attribute*

```python
volume: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1039-L1039)

### particula.gpu.kernels.exhaustion._PreparedResamplingCall.__init__

*method*

```python
def __init__(result: Any, masses: Any, concentration: Any, charge: Any, density: Any, volume: Any, required_release_counts: Any, buffer_arrays: tuple[Any, ...], bounds: tuple[float, float, float, float], dimensions: tuple[int, int, int], device: Any, invalid: Any, planning_failed: Any, upstream_invalid: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L0-L0)

## particula.gpu.kernels.exhaustion._aggregate_planning_status

*function*

```python
def _aggregate_planning_status(counts: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32), result: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32), upstream_invalid: wp.array(dtype=wp.int32)) -> None
```

Record whether any nonzero-demand box failed planning.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L736-L752)

## particula.gpu.kernels.exhaustion._apply_representative_volume_scaling

*function*

```python
def _apply_representative_volume_scaling(concentration: wp.array2d(dtype=wp.float64), volume: wp.array(dtype=wp.float64), demand: wp.array(dtype=wp.float64), selected: wp.array(dtype=wp.int32), requested_scale: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32), upstream_invalid: wp.array(dtype=wp.int32)) -> None
```

Scale only selected volume, concentration, and demand storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L876-L893)

## particula.gpu.kernels.exhaustion._commit_resampling

*function*

```python
def _commit_resampling(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), counts: wp.array(dtype=wp.int32), retained_counts: wp.array(dtype=wp.int32), retained_indices: wp.array2d(dtype=wp.int32), released_indices: wp.array2d(dtype=wp.int32), replacement_masses: wp.array3d(dtype=wp.float64), replacement_concentration: wp.array2d(dtype=wp.float64), replacement_charge: wp.array2d(dtype=wp.float64), invalid: wp.array(dtype=wp.int32), planning_failed: wp.array(dtype=wp.int32), upstream_invalid: wp.array(dtype=wp.int32)) -> None
```

Commit all successful box plans in one fixed-shape launch.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L693-L733)

## particula.gpu.kernels.exhaustion._enqueue_prepared_representative_volume_scaling_call

*function*

```python
def _enqueue_prepared_representative_volume_scaling_call(prepared: _PreparedRepresentativeVolumeScalingCall) -> tuple[Any, Any, Any]
```

Enqueue one fixed scaling sequence using only pinned arrays.

This phase performs no allocation, host readback, synchronization, or
container lookup. Rollback is not promised after a writer launch.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1535-L1585)

## particula.gpu.kernels.exhaustion._enqueue_prepared_resampling_call

*function*

```python
def _enqueue_prepared_resampling_call(prepared: _PreparedResamplingCall) -> Any
```

Enqueue one fixed resampling sequence using only pinned arrays.

This phase performs no allocation, host readback, synchronization, or
container lookup. A writer launch has the primitive's no-rollback boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1285-L1346)

## particula.gpu.kernels.exhaustion._enqueue_resampling_validation

*function*

```python
def _enqueue_resampling_validation(prepared: _PreparedResamplingCall) -> None
```

Reset and enqueue the complete read-only resampling validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1099-L1165)

## particula.gpu.kernels.exhaustion._enqueue_scaling_validation

*function*

```python
def _enqueue_scaling_validation(prepared: _PreparedRepresentativeVolumeScalingCall) -> None
```

Reset and enqueue complete read-only representative validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1482-L1532)

## particula.gpu.kernels.exhaustion._field

*function*

```python
def _field(container: Any, name: str) -> Any
```

Return a required particle field with a stable validation error.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L924-L929)

## particula.gpu.kernels.exhaustion._lane_read

*function*

```python
def _lane_read(primary: wp.array2d(dtype=wp.int32), secondary: wp.array2d(dtype=wp.int32), box: Any, lane: Any, primary_size: Any) -> int
```

Read one logical sort lane from the combined index workspace.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L268-L279)

## particula.gpu.kernels.exhaustion._lane_write

*function*

```python
def _lane_write(primary: wp.array2d(dtype=wp.int32), secondary: wp.array2d(dtype=wp.int32), box: Any, lane: Any, primary_size: Any, value: Any) -> None
```

Write one logical sort lane into the combined index workspace.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L282-L295)

## particula.gpu.kernels.exhaustion._less_source

*function*

```python
def _less_source(masses: wp.array3d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), source_radii: wp.array2d(dtype=wp.float64), box: Any, left: Any, right: Any, species_count: Any) -> bool
```

Compare the CPU resampling key, including original index tie breaking.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L141-L172)

## particula.gpu.kernels.exhaustion._observe_prepared_representative_volume_scaling_call

*function*

```python
def _observe_prepared_representative_volume_scaling_call(prepared: _PreparedRepresentativeVolumeScalingCall) -> tuple[Any, Any, Any]
```

Observe scaling status and return the original caller identities.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1588-L1596)

## particula.gpu.kernels.exhaustion._observe_prepared_resampling_call

*function*

```python
def _observe_prepared_resampling_call(prepared: _PreparedResamplingCall) -> Any
```

Observe resampling status and return the original particle carrier.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1349-L1359)

## particula.gpu.kernels.exhaustion._plan_resampling

*function*

```python
def _plan_resampling(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), density: wp.array(dtype=wp.float64), counts: wp.array(dtype=wp.int32), retained_counts: wp.array(dtype=wp.int32), released_counts: wp.array(dtype=wp.int32), retained_indices: wp.array2d(dtype=wp.int32), released_indices: wp.array2d(dtype=wp.int32), sorted_indices: wp.array2d(dtype=wp.int32), replacement_masses: wp.array3d(dtype=wp.float64), replacement_concentration: wp.array2d(dtype=wp.float64), replacement_charge: wp.array2d(dtype=wp.float64), source_radii: wp.array2d(dtype=wp.float64), radius_error: wp.array(dtype=wp.float64), mean_error: wp.array(dtype=wp.float64), surface_error: wp.array(dtype=wp.float64), diversity_error: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32), upstream_invalid: wp.array(dtype=wp.int32), radius_bound: wp.float64, mean_bound: wp.float64, surface_bound: wp.float64, diversity_bound: wp.float64) -> None
```

Plan one box sequentially; output rows remain caller-owned storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L378-L690)

## particula.gpu.kernels.exhaustion._prepare_representative_volume_scaling_step_gpu

*function*

```python
def _prepare_representative_volume_scaling_step_gpu(particles: Any, provisional_source_demand: Any, scaling_required: Any, requested_scale: Any, minimum_scale: Any, minimum_volume: Any, resolved_scale: Any, *, _upstream_invalid: Any | None = None) -> _PreparedRepresentativeVolumeScalingCall
```

Validate metadata, allocate, and pin one private scaling invocation.

This private setup phase validates fixed schemas but leaves mutable payload
validation to enqueue and bounded status observation to the observer.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1388-L1479)

## particula.gpu.kernels.exhaustion._prepare_resampling_step_gpu

*function*

```python
def _prepare_resampling_step_gpu(particles: Any, required_release_counts: Any, buffers: ResamplingBuffers, *, radius_cubed_relative_error: float = 1.0, mean_radius_relative_error: float = 1.0, surface_relative_error: float = 1.0, diversity_absolute_error: float = 1.0, _upstream_invalid: Any | None = None) -> _PreparedResamplingCall
```

Validate metadata, allocate, and pin one private resampling invocation.

This private setup phase validates fixed schemas and controls but does not
inspect mutable payload values. Enqueue owns device validation and all
writes; observation is the sole status readback.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1168-L1282)

## particula.gpu.kernels.exhaustion._radius

*function*

```python
def _radius(masses: wp.array3d(dtype=wp.float64), density: wp.array(dtype=wp.float64), box: Any, particle: Any, species_count: Any) -> wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L124-L138)

## particula.gpu.kernels.exhaustion._ranges_overlap

*function*

```python
def _ranges_overlap(left: Any, right: Any) -> bool
```

Return whether two caller allocations have overlapping byte ranges.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L969-L980)

## particula.gpu.kernels.exhaustion._require_contiguous_array

*function*

```python
def _require_contiguous_array(value: Any, name: str) -> None
```

Reject strided Warp views before pointer-range ownership checks.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L954-L966)

## particula.gpu.kernels.exhaustion._require_exact_bound

*function*

```python
def _require_exact_bound(name: str, value: Any) -> float
```

Return a finite nonnegative exact Python-float diagnostic bound.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L915-L921)

## particula.gpu.kernels.exhaustion._reset_prepared_scaling_status

*function*

```python
def _reset_prepared_scaling_status(status: wp.array(dtype=wp.int32)) -> None
```

Reset the invalid and selected scaling status lanes on device.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L906-L912)

## particula.gpu.kernels.exhaustion._reset_prepared_status

*function*

```python
def _reset_prepared_status(first: wp.array(dtype=wp.int32), second: wp.array(dtype=wp.int32)) -> None
```

Reset two preparation-owned scalar status arrays on device.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L896-L903)

## particula.gpu.kernels.exhaustion._resolve_representative_volume_selection

*function*

```python
def _resolve_representative_volume_selection(demand: wp.array(dtype=wp.float64), scaling_required: wp.array(dtype=wp.int32), selected: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32), upstream_invalid: wp.array(dtype=wp.int32)) -> None
```

Capture the immutable selected-row predicate before any writes.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L837-L855)

## particula.gpu.kernels.exhaustion._riemer_diversity_dense

*function*

```python
def _riemer_diversity_dense(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), box: Any, count: Any, species_count: Any) -> wp.float64
```

Compute Riemer diversity over a dense replacement prefix.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L223-L265)

## particula.gpu.kernels.exhaustion._riemer_diversity_scattered

*function*

```python
def _riemer_diversity_scattered(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), indices: wp.array2d(dtype=wp.int32), box: Any, count: Any, species_count: Any) -> wp.float64
```

Compute Riemer diversity over an indexed active source sequence.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L175-L220)

## particula.gpu.kernels.exhaustion._scan_density

*function*

```python
def _scan_density(density: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Validate global density when a batch has no particle rows.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L329-L337)

## particula.gpu.kernels.exhaustion._scan_density_p4

*function*

```python
def _scan_density_p4(density: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32)) -> None
```

Validate global density state, including empty particle-box inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L826-L834)

## particula.gpu.kernels.exhaustion._scan_particle_values

*function*

```python
def _scan_particle_values(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), density: wp.array(dtype=wp.float64), volume: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Perform one read-only physical-state validation scan.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L298-L326)

## particula.gpu.kernels.exhaustion._scan_representative_volume_scaling

*function*

```python
def _scan_representative_volume_scaling(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), density: wp.array(dtype=wp.float64), volume: wp.array(dtype=wp.float64), demand: wp.array(dtype=wp.float64), scaling_required: wp.array(dtype=wp.int32), requested_scale: wp.array(dtype=wp.float64), minimum_scale: wp.array(dtype=wp.float64), minimum_volume: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32)) -> None
```

Check physical and volume scaling sidecars in one read-only scan.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L755-L823)

## particula.gpu.kernels.exhaustion._scan_slot_state

*function*

```python
def _scan_slot_state(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), density: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Validate inactive literal-zero state and active material volume.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L340-L359)

## particula.gpu.kernels.exhaustion._validate_array

*function*

```python
def _validate_array(value: Any, name: str, rank: int, shape: tuple[int, ...], dtype: Any, device: Any) -> None
```

Validate caller-owned Warp metadata without reading array values.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L932-L951)

## particula.gpu.kernels.exhaustion._validate_buffers

*function*

```python
def _validate_buffers(buffers: ResamplingBuffers, b: int, n: int, s: int, device: Any, protected: list[Any]) -> None
```

Validate buffer metadata and enforce distinct non-aliasing storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L983-L1019)

## particula.gpu.kernels.exhaustion._validate_counts

*function*

```python
def _validate_counts(concentration: wp.array2d(dtype=wp.float64), counts: wp.array(dtype=wp.int32), invalid: wp.array(dtype=wp.int32)) -> None
```

Validate each resolved nonzero release demand against active capacity.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L362-L375)

## particula.gpu.kernels.exhaustion._validate_prepared_upstream

*function*

```python
def _validate_prepared_upstream(upstream_invalid: Any | None, device: Any) -> Any
```

Return a same-device scalar gate, allocating the standalone gate.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1081-L1096)

## particula.gpu.kernels.exhaustion._write_resolved_representative_scale

*function*

```python
def _write_resolved_representative_scale(selected: wp.array(dtype=wp.int32), requested_scale: wp.array(dtype=wp.float64), resolved_scale: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32), upstream_invalid: wp.array(dtype=wp.int32)) -> None
```

Write the volume scaling diagnostic after read-only preflight.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L858-L873)

## particula.gpu.kernels.exhaustion.representative_volume_scaling_step_gpu

*function*

```python
def representative_volume_scaling_step_gpu(particles: Any, provisional_source_demand: Any, scaling_required: Any, requested_scale: Any, minimum_scale: Any, minimum_volume: Any, resolved_scale: Any) -> tuple[Any, Any, Any]
```

Scale selected Warp particle-container volumes in place.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1599-L1619)

## particula.gpu.kernels.exhaustion.resampling_step_gpu

*function*

```python
def resampling_step_gpu(particles: Any, required_release_counts: Any, buffers: ResamplingBuffers, *, radius_cubed_relative_error: float = 1.0, mean_radius_relative_error: float = 1.0, surface_relative_error: float = 1.0, diversity_absolute_error: float = 1.0) -> Any
```

Apply deterministic fixed-capacity resampling equal-weight remapping in
place.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/exhaustion.py#L1362-L1385)
