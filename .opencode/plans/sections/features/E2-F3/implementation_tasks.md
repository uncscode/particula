# Implementation Tasks

## Preparation

- [ ] Confirm `EnvironmentData` exists from `E2-F2` and identify its import path.
- [ ] Confirm every GPU-transferable field is a numeric NumPy array shaped
      `(n_boxes,)`.
- [ ] If `E2-F2` has not landed, block this feature or include the minimal CPU
      schema first; do not invent a conflicting schema.

## Warp struct

- [ ] Add `WarpEnvironmentData` to `particula/gpu/warp_types.py`.
- [ ] Add docstring field documentation and shape conventions.
- [ ] Add import/export coverage in `particula/gpu/tests/warp_types_test.py`.

## Conversion helpers

- [ ] Add `to_warp_environment_data` using `_ensure_warp_available` and
      `_validate_device`.
- [ ] Transfer each field explicitly; avoid loops that hide missing fields from
      reviewers unless they are generated from a documented schema tuple.
- [ ] Add `from_warp_environment_data` with optional `sync` and exact CPU
      reconstruction.
- [ ] Update `particula/gpu/__init__.py` imports and `__all__`.

## Tests

- [ ] Add fixtures for one-box and multi-box `EnvironmentData` instances.
- [ ] Add CPU-device transfer tests for values, shapes, and dtypes.
- [ ] Add round-trip tests comparing every field with NumPy assertions.
- [ ] Add CUDA-parametrized coverage using `warp_devices(wp)`.
- [ ] Add tests proving no conversion happens outside explicit helper calls.

## Documentation

- [ ] Update `docs/Features/Roadmap/data-oriented-gpu.md` progress notes.
- [ ] Update `docs/Features/particle-data-migration.md` with environment
      transfer examples.
- [ ] Update `docs/Theory/nvidia-warp/datastructures.md` with the new struct.
