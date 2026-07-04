# Architecture Design

## Design summary

`WarpEnvironmentData` is a thin Warp struct that mirrors the CPU
`EnvironmentData` schema from `E2-F2`. The landed implementation stores only
numeric per-box state directly on the struct and keeps this phase limited to
schema declaration plus test coverage. Conversion helpers remain a later-phase
boundary and were not added in issue `#1192`.

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
3. `particula/gpu/tests/warp_types_test.py` verifies field names, shapes,
   dtypes, and deterministic stored values.
4. Later `E2-F3` phases can add explicit transfer helpers on top of this stable
   schema boundary.

## Boundary principles

- No conversion helper should be called implicitly by kernels, runnables, or
  strategy objects.
- This phase intentionally avoids helper APIs so there is no hidden transfer
  behavior to document or validate yet.
- Existing scalar kernel APIs remain stable until later migration tracks.

## Compatibility

The struct follows current `ParticleData` and `GasData` Warp declaration
patterns, so later transfer helpers can align with existing GPU APIs without
revisiting the field schema.
