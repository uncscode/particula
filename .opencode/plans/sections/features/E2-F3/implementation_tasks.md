# Implementation Tasks

## Preparation

- [ ] Confirm `EnvironmentData` exists from `E2-F2` and identify its import path.
- [ ] Confirm the CPU field contract is exactly `temperature`, `pressure`, and
      `saturation_ratio`, with shapes `(n_boxes,)`, `(n_boxes,)`, and
      `(n_boxes, n_species)` before touching GPU code.
- [ ] If `E2-F2` has not landed, block this feature or include the minimal CPU
      schema first; do not invent a conflicting schema.

## Warp struct

- [ ] Add `WarpEnvironmentData` to `particula/gpu/warp_types.py`.
- [ ] Declare explicit `temperature`, `pressure`, and `saturation_ratio`
      attributes in `WarpEnvironmentData` instead of a generic loop or hidden
      schema expansion.
- [ ] Add docstring field documentation describing `(n_boxes,)` scalar fields
      and `(n_boxes, n_species)` saturation-ratio storage.
- [ ] Add import/export and shape coverage in
      `particula/gpu/tests/warp_types_test.py`, including one-box and multi-box
      construction assertions.

## Conversion helpers

- [x] Add `to_warp_environment_data` using `_ensure_warp_available` and
      `_validate_device`.
- [x] Transfer `temperature`, `pressure`, and `saturation_ratio` explicitly in
      `particula/gpu/conversion.py`; avoid loops that would hide field drift from
      reviewers unless they are generated from a documented shared schema tuple.
- [ ] Add `from_warp_environment_data` with optional `sync` and exact CPU
      reconstruction.
- [ ] Update `particula/gpu/__init__.py` imports and `__all__`.

## Tests

- [x] Add fixtures for one-box and multi-box `EnvironmentData` instances.
- [x] Add CPU-device transfer tests in
      `particula/gpu/tests/conversion_test.py` for values, shapes, and dtypes of
      all three fields.
- [x] Add invalid-device, Warp-unavailable, and CPU copy-semantics coverage for
      `to_warp_environment_data` in `particula/gpu/tests/conversion_test.py`.
- [ ] Add round-trip tests comparing `temperature`, `pressure`, and
      `saturation_ratio` with NumPy assertions and explicit `sync=True`/
      `sync=False` coverage.
- [ ] Add CUDA-parametrized coverage using `warp_devices(wp)`.
- [ ] Add tests proving no conversion happens outside explicit helper calls.

## Documentation

- [ ] Update `docs/Features/Roadmap/data-oriented-gpu.md` progress notes with
      the exact helper names `to_warp_environment_data` and
      `from_warp_environment_data`.
- [ ] Update `docs/Features/particle-data-migration.md` with environment
      transfer examples.
- [ ] Update `docs/Theory/nvidia-warp/datastructures.md` with the new struct.
