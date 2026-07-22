# Implementation Tasks

## GPU Kernels and Entry Point

- [x] Freeze the concrete-only `dilution_step_gpu(particles, gas, coefficient, time_step)` signature and scalar/`(n_boxes,)` coefficient metadata contract in `particula/gpu/kernels/dilution.py` from E6-F1 behavior (P1, #1395).
- [ ] Add Warp kernels that update particle concentration `(n_boxes, n_particles)` and gas concentration `(n_boxes, n_species)` by the same per-box factor.
- [ ] Preserve fixed shapes, devices, dtypes, object/array identity, inactive slots, and every nondiluted field.
- [ ] Implement exact zero-time and zero-coefficient fast paths without writes.
- [ ] Re-export only the low-level step from `particula/gpu/kernels/__init__.py` (P2; intentionally not done in P1).

## Validation

- [ ] Validate container ranks, cross-field dimensions, `wp.float64` concentration arrays, and same-device ownership before launch.
- [x] Validate scalar coefficient and time forms/domains and per-box coefficient rank, shape, dtype, and device metadata without a caller-state write (P1, #1395).
- [ ] Validate per-box coefficient values and finite nonnegative concentrations (P3).
- [ ] Establish rejection ordering before private scalar-buffer allocation and before particle or gas mutation.
- [ ] Reject NumPy/host arrays at the process boundary instead of transferring or falling back.

## Tooling / Tests

- [x] Add `particula/gpu/kernels/tests/dilution_test.py` with P1 signature/import, normalization, identity/no-write, zero-box, and rejection-order coverage (#1395).
- [ ] Cover scalar and nonuniform per-box inputs, one/multiple boxes and species, inactive slots, repeated calls, and zero inputs.
- [ ] Snapshot all caller-owned arrays for invalid-call atomicity and protected-field invariants.
- [ ] Require Warp CPU parity and skip optional CUDA evidence cleanly when unavailable.
- [ ] Add direct import/API smoke tests and focused `pytest ... -q -Werror` commands.
