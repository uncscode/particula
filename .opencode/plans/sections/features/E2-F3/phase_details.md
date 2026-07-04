# Phase Details

## E2-F3-P1: WarpEnvironmentData struct and CPU schema alignment with unit tests

- Issue: TBD | Size: S | Status: Not Started
- Goal: Mirror the `E2-F2` CPU environment schema in Warp with a small, reviewable
  struct change and tests that prove every field stays `(n_boxes,)`.
- Files: `particula/gpu/warp_types.py`,
  `particula/gpu/tests/warp_types_test.py`
- Implementation:
  - Confirm the `E2-F2` `EnvironmentData` import path and exact field names.
  - Add `WarpEnvironmentData` to `particula/gpu/warp_types.py` with one
    `wp.array(dtype=wp.float64)` attribute per numeric CPU field.
  - Keep the struct edit focused to field declarations/docstrings rather than a
    broad refactor of existing Warp types.
- Tests: Add one-box and multi-box struct creation coverage plus dtype/shape
  assertions in `particula/gpu/tests/warp_types_test.py`.

## E2-F3-P2: CPU-to-Warp environment conversion helper with unit tests

- Issue: TBD | Size: S | Status: Not Started
- Goal: Add a single explicit conversion helper that reviewers can trace field by
  field from `EnvironmentData` into `WarpEnvironmentData`.
- Files: `particula/gpu/conversion.py`,
  `particula/gpu/tests/conversion_test.py`
- Implementation:
  - Add `to_warp_environment_data(data, device="cuda", copy=True)` to
    `particula/gpu/conversion.py`.
  - Reuse `_ensure_warp_available` and `_validate_device` rather than adding a
    second device-validation path.
  - Transfer each field explicitly with `wp.float64` buffers so missing fields
    stay obvious in code review.
- Tests: Add CPU-device tests for values, shapes, dtypes, invalid devices, and
  `copy=True` versus `copy=False` behavior where Warp supports host aliasing.

## E2-F3-P3: Warp-to-CPU environment conversion and round-trip coverage

- Issue: TBD | Size: S | Status: Not Started
- Goal: Reconstruct the CPU container with no hidden field dropping so round-trip
  expectations are executable in tests.
- Files: `particula/gpu/conversion.py`, `particula/gpu/__init__.py`,
  `particula/gpu/tests/conversion_test.py`
- Implementation:
  - Add `from_warp_environment_data(gpu_data, sync=True)` to
    `particula/gpu/conversion.py`.
  - Rebuild `EnvironmentData` from `.numpy()` arrays in declared field order.
  - Export `WarpEnvironmentData`, `to_warp_environment_data`, and
    `from_warp_environment_data` from `particula/gpu/__init__.py`.
- Tests: Add `sync=True`, `sync=False`, CPU-backend round-trip, and multi-box
  equality coverage for every environment field.

## E2-F3-P4: CUDA-parametrized transfer coverage and documentation updates

- Issue: TBD | Size: XS | Status: Not Started
- Goal: Close the feature with optional CUDA parity coverage and docs that point
  users to the exact transfer entry points.
- Files: `particula/gpu/tests/conversion_test.py`,
  `docs/Features/Roadmap/data-oriented-gpu.md`,
  `docs/Features/particle-data-migration.md`,
  `docs/Theory/nvidia-warp/datastructures.md`
- Implementation:
  - Use `warp_devices(wp)` to run the core round-trip test on `cpu` and `cuda`
    when CUDA is available.
  - Keep CUDA coverage optional and skip cleanly on CPU-only machines.
  - Update roadmap, migration, and Warp datastructure docs with the explicit
    environment-transfer helper names and shape semantics.
- Tests: Keep the CUDA-parametrized test focused to one representative round trip
  so this phase stays XS-sized.
