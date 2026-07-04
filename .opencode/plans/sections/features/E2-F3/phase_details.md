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
landed next in `#1194`.

## E2-F3-P3: Warp-to-CPU environment conversion and round-trip coverage

- Issue: #1194 | Size: S | Status: Shipped
- Goal: Reconstruct the CPU container with no hidden field dropping so round-trip
  expectations are executable in tests.
- Files: `particula/gpu/conversion.py`, `particula/gpu/__init__.py`,
  `particula/gpu/tests/conversion_test.py`,
  `particula/gpu/tests/warp_types_test.py`, `particula/gas/environment_data.py`
- Implementation:
  - Added `from_warp_environment_data(gpu_data, sync=True)` to
    `particula/gpu/conversion.py` using `_ensure_warp_available()` and the
    same explicit field-by-field reconstruction style as the other reverse
    helpers.
  - Rebuilt `EnvironmentData` from `.numpy()` arrays in declared field order so
    CPU-side constructor validation remains the schema check.
  - Exported `WarpEnvironmentData`, `to_warp_environment_data`, and
    `from_warp_environment_data` from `particula/gpu/__init__.py`.
  - Updated the `EnvironmentData` module docstring so it no longer claims the
    environment container lacks CPU↔GPU helpers.
- Tests: Added `sync=True`, manual `wp.synchronize()` plus `sync=False`,
  single-box and multi-box CPU round trips, malformed-schema failure coverage
  in `particula/gpu/tests/conversion_test.py`, and public-export assertions in
  `particula/gpu/tests/warp_types_test.py`.

Status note: Shipped in issue `#1194`. The public environment round-trip helper
surface is now complete on the Warp CPU backend.

## E2-F3-P4: CUDA-parametrized transfer coverage and documentation updates

- Issue: #1195 | Size: S | Status: Shipped
- Goal: Close the feature with optional CUDA parity coverage and docs that point
  users to the exact transfer entry points.
- Files: `particula/gpu/tests/conversion_test.py`,
  `docs/Features/Roadmap/data-oriented-gpu.md`,
  `docs/Features/particle-data-migration.md`,
  `docs/Theory/nvidia-warp/datastructures.md`
- Implementation:
  - Added `from particula.gpu.tests.cuda_availability import warp_devices` to
    `particula/gpu/tests/conversion_test.py` and used
    `@pytest.mark.parametrize("device", warp_devices(wp))` for one focused
    environment round-trip parity test.
  - Reused the existing multi-box fixture so the new test proves exact value
    and shape preservation for `(n_boxes,)` `temperature`/`pressure` and
    `(n_boxes, n_species)` `saturation_ratio` on Warp CPU and CUDA when
    available.
  - Updated `docs/Features/particle-data-migration.md` and
    `docs/Features/Roadmap/data-oriented-gpu.md` to state that environment
    transfers only happen through `particula.gpu.WarpEnvironmentData`,
    `to_warp_environment_data()`, and `from_warp_environment_data()`.
  - Added a concrete environment round-trip example to
    `docs/Theory/nvidia-warp/datastructures.md` and documented the field-shape
    rules there.
- Tests: The new representative parity case stays intentionally narrow while
  extending existing CPU-only round-trip, `sync=False`, and malformed-shape
  coverage already present in `particula/gpu/tests/conversion_test.py`.

Status note: Shipped in issue `#1195`. `E2-F3` now has explicit helper docs,
shape-contract docs, and one device-aware parity test for CPU plus optional
CUDA coverage.
