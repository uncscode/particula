# Implementation Tasks

## Preparation

- [x] Confirm `EnvironmentData` exists from `E2-F2` and identify its import path.
- [x] Confirm the CPU field contract is exactly `temperature`, `pressure`, and
      `saturation_ratio`, with shapes `(n_boxes,)`, `(n_boxes,)`, and
      `(n_boxes, n_species)` before touching GPU code.
- [ ] If `E2-F2` has not landed, block this feature or include the minimal CPU
      schema first; do not invent a conflicting schema.

## Warp struct

- [x] Add `WarpEnvironmentData` to `particula/gpu/warp_types.py`.
- [x] Declare explicit `temperature`, `pressure`, and `saturation_ratio`
      attributes in `WarpEnvironmentData` instead of a generic loop or hidden
      schema expansion.
- [x] Add docstring field documentation describing `(n_boxes,)` scalar fields
      and `(n_boxes, n_species)` saturation-ratio storage.
- [x] Add import/export and shape coverage in
      `particula/gpu/tests/warp_types_test.py`, including one-box and multi-box
      construction assertions.

## Conversion helpers

- [x] Add `to_warp_environment_data` using `_ensure_warp_available` and
      `_validate_device`.
- [x] Transfer `temperature`, `pressure`, and `saturation_ratio` explicitly in
      `particula/gpu/conversion.py`; avoid loops that would hide field drift from
      reviewers unless they are generated from a documented shared schema tuple.
- [x] Add `from_warp_environment_data` with optional `sync` and exact CPU
      reconstruction.
- [x] Update `particula/gpu/__init__.py` imports and `__all__`.

## Tests

- [x] Add fixtures for one-box and multi-box `EnvironmentData` instances.
- [x] Add CPU-device transfer tests in
      `particula/gpu/tests/conversion_test.py` for values, shapes, and dtypes of
      all three fields.
- [x] Add invalid-device, Warp-unavailable, and CPU copy-semantics coverage for
      `to_warp_environment_data` in `particula/gpu/tests/conversion_test.py`.
- [x] Add round-trip tests comparing `temperature`, `pressure`, and
      `saturation_ratio` with NumPy assertions and explicit `sync=True`/
      `sync=False` coverage.
- [x] Add CUDA-parametrized coverage using `warp_devices(wp)`.
- [x] Add tests proving malformed environment reconstruction fails at the
      explicit helper boundary rather than via hidden conversion paths.

## Documentation

- [x] Update `docs/Features/Roadmap/data-oriented-gpu.md` progress notes with
      the exact helper names `to_warp_environment_data` and
      `from_warp_environment_data`.
- [x] Update `docs/Features/particle-data-migration.md` with environment
      transfer examples.
- [x] Update `docs/Theory/nvidia-warp/datastructures.md` with the new struct.
