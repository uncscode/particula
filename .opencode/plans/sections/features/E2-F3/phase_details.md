# Phase Details

## E2-F3-P1: WarpEnvironmentData struct and CPU schema alignment with unit tests

- Issue: #1192 | Size: S | Status: Shipped
- Goal: Mirror the `E2-F2` CPU environment schema in Warp with a small,
  reviewable struct change and tests that prove `temperature` and `pressure`
  stay `(n_boxes,)` while `saturation_ratio` stays `(n_boxes, n_species)`.
- Files: `particula/gpu/warp_types.py`,
  `particula/gpu/tests/warp_types_test.py`
- Implementation:
  - Confirm the `E2-F2` `EnvironmentData` import path and exact field names.
  - Add `WarpEnvironmentData` to `particula/gpu/warp_types.py` with explicit
    `temperature` and `pressure` `wp.array(dtype=wp.float64)` fields plus a
    `saturation_ratio` `wp.array2d(dtype=wp.float64)` field matching the CPU
    field list and shapes.
  - Expand the module docstring and add a dedicated `WarpEnvironmentData`
    class docstring describing the environment schema.
  - Keep the struct edit focused to field declarations/docstrings rather than a
    broad refactor of existing Warp types.
- Tests: Added one-box and multi-box struct creation coverage plus dtype/shape
  assertions, field-access checks, and deterministic NumPy-backed round-trip
  value assertions in `particula/gpu/tests/warp_types_test.py`.

## E2-F3-P2: CPU-to-Warp environment conversion helper with unit tests

- Issue: #1193 | Size: S | Status: Shipped
- Goal: Add a single explicit conversion helper that reviewers can trace field by
  field from `EnvironmentData` into `WarpEnvironmentData`.
- Files: `particula/gpu/conversion.py`,
  `particula/gpu/tests/conversion_test.py`
- Implementation:
  - Added `to_warp_environment_data(data, device="cuda", copy=True)` to
    `particula/gpu/conversion.py` immediately alongside the existing particle
    and gas helpers.
  - Reused `_ensure_warp_available` and `_validate_device` so missing-Warp and
    invalid-device behavior follows the established helper path.
  - Transferred `temperature`, `pressure`, and `saturation_ratio` explicitly
    with `wp.float64` arrays for both `copy=True` (`wp.array`) and
    `copy=False` (`wp.from_numpy`) paths.
  - Added a Google-style helper docstring documenting arguments, return type,
    and `RuntimeError` failure modes.
- Tests: Added CPU-device tests for populated fields, single-box and multi-box
  values/shapes, `wp.float64` dtypes, invalid devices, Warp-unavailable
  behavior, and `copy=True` versus conservative CPU `copy=False` semantics in
  `particula/gpu/tests/conversion_test.py`.

Status note: Shipped in issue `#1193`. Reverse conversion and public exports
remain future-phase work.

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

Status note: Not started. No package exports or CPU reconstruction helpers have
been added yet.

## E2-F3-P4: CUDA-parametrized transfer coverage and documentation updates

- Issue: TBD | Size: S | Status: Not Started
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
  so the `S`-sized phase remains reviewable.
  The structured phase metadata is authoritative: this phase is `S` because it
  combines optional CUDA transfer coverage with user-facing documentation updates.

Status note: Not started. No CUDA transfer coverage or external docs were
required for the shipped P1/P2 implementation slices.
