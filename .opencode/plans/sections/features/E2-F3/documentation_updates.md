# Documentation Updates

## Required updates

- `docs/Features/Roadmap/data-oriented-gpu.md`: mark the environment Warp
  mirror and conversion helpers as implemented or in progress.
- `docs/Features/particle-data-migration.md`: add a short example showing
  `EnvironmentData` -> `WarpEnvironmentData` -> `EnvironmentData` round trip.
- `docs/Theory/nvidia-warp/datastructures.md`: document the new struct fields,
  `(n_boxes,)` convention, and device ownership.

## Content to include

- Transfer helpers are explicit API calls; no hidden transfers occur in kernels
  or runnable objects.
- Default helper device remains aligned with existing GPU helpers, while tests
  use `device="cpu"` for portability.
- CUDA test coverage is conditional on Warp reporting CUDA availability.
- Environment string metadata, if any is added later, remains CPU-only unless a
  separate GPU representation is designed.

## Documentation examples

```python
gpu_environment = to_warp_environment_data(
    environment,
    device="cpu",
)
round_trip = from_warp_environment_data(gpu_environment)
```

The example should note that production GPU runs may pass `device="cuda"` when
Warp and CUDA are available.
