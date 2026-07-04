# Architecture Design

## Design summary

`WarpEnvironmentData` is a thin Warp struct that mirrors the CPU
`EnvironmentData` schema from `E2-F2`. The landed implementation stores only
numeric per-box state directly on the struct. The landed work now includes both
explicit transfer directions, `to_warp_environment_data(...)` and
`from_warp_environment_data(...)`, plus public `particula.gpu` exports for the
environment helper surface.

## Proposed API

```python
@wp.struct
class WarpEnvironmentData:
    temperature: wp.array(dtype=wp.float64)
    pressure: wp.array(dtype=wp.float64)
    saturation_ratio: wp.array2d(dtype=wp.float64)
```

`WarpEnvironmentData` mirrors the CPU `EnvironmentData` schema exactly:
`temperature` and `pressure` are shaped `(n_boxes,)`, and `saturation_ratio` is
shaped `(n_boxes, n_species)` with finite nonnegative supersaturation values
allowed. It does not add simulation volume; `ParticleData.volume` remains the
authoritative volume carrier.

## Data flow

1. CPU schema decisions come from `E2-F2` `EnvironmentData`.
2. `E2-F3-P1` declares the matching Warp struct in
   `particula/gpu/warp_types.py`.
3. `E2-F3-P2` adds `to_warp_environment_data` in
   `particula/gpu/conversion.py` using `_ensure_warp_available`,
   `_validate_device`, and explicit field-by-field assignment.
4. `E2-F3-P3` adds `from_warp_environment_data` in
   `particula/gpu/conversion.py`, reconstructing `EnvironmentData` from the
   three explicit Warp fields after optional synchronization.
5. `particula/gpu/tests/warp_types_test.py` and
   `particula/gpu/tests/conversion_test.py` verify field names, shapes, dtypes,
   values, helper behavior, public exports, sync paths, and schema failures.
6. Later `E2-F3` phases can add optional CUDA parity and higher-level runtime
   integration on top of this stable schema boundary.

## Boundary principles

- No conversion helper should be called implicitly by kernels, runnables, or
  strategy objects.
- Environment transfers now have explicit helper APIs in both directions, but
  hidden transfer behavior is still disallowed.
- Existing scalar kernel APIs remain stable until later migration tracks.

## Compatibility

The struct follows current `ParticleData` and `GasData` Warp declaration
patterns, so later transfer helpers can align with existing GPU APIs without
revisiting the field schema.
