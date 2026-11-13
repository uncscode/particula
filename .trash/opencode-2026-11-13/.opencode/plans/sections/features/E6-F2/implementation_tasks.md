# Implementation Tasks

## GPU Kernels and Entry Point

- [x] Freeze the concrete-only `dilution_step_gpu(particles, gas, coefficient, time_step)` signature and scalar/`(n_boxes,)` coefficient metadata contract in `particula/gpu/kernels/dilution.py` from E6-F1 behavior (P1, #1395).
- [x] Add Warp kernels that update particle concentration `(n_boxes, n_particles)` and gas concentration `(n_boxes, n_species)` by the same per-box factor (P2, #1396).
- [x] Preserve fixed shapes, devices, dtypes, object/array identity, inactive slots, and every nondiluted field (P2, #1396).
- [x] Implement exact zero-time and zero-coefficient fast paths without writes (P2, #1396).
- [x] Re-export only the low-level step from `particula/gpu/kernels/__init__.py` (P2, #1396).

## Validation

- [x] Validate launch-safety concentration ranks, box dimensions, `wp.float64` arrays, and same-device ownership before non-no-op launches (P2, #1396).
- [x] Validate scalar coefficient and time forms/domains and per-box coefficient rank, shape, dtype, and device metadata without a caller-state write (P1, #1395).
- [x] Validate per-box coefficient values and finite nonnegative concentrations (P3, #1397).
- [x] Establish documented rejection ordering before private scalar-buffer allocation and before particle or gas mutation (P3, #1397).
- [x] Reject NumPy/host arrays at the process boundary instead of transferring or falling back (P3, #1397).

## Tooling / Tests

- [x] Add `particula/gpu/kernels/tests/dilution_test.py` with P1 signature/import, normalization, identity/no-write, zero-box, and rejection-order coverage (#1395).
- [x] Cover scalar and nonuniform per-box inputs, one/multiple boxes and species, inactive slots, repeated calls, and zero inputs (P2, #1396).
- [x] Snapshot protected fields and caller-owned identities for P2 execution and metadata failures (#1396); full invalid-call atomicity remains P3.
- [x] Add P3 snapshots, precedence cases, and allocation/launch spies for rejected and valid no-op calls (#1397).
- [x] Require Warp CPU finite-step parity and run the identical optional CUDA
  matrix with a clean unavailable-CUDA skip; cover independent NumPy-reference
  scalar/per-box, one/multi-box, multi-species, repeated-call, invariant, and
  exact no-op cases (P4, #1398).
- [x] Add direct package-import/API smoke tests and the focused `pytest ... -q -Werror` command (P2, #1396).

## Documentation and Publication

- [x] Publish the P1–P4 contract in the foundation guide, GPU roadmap, and
  `AGENTS.md`, including caller ownership, preflight/no-op boundaries, tolerances,
  optional CUDA, and deferred functionality (P5, #1399).
- [x] Reconcile README and documentation indexes with the direct-kernel guidance
  (P5, #1399).
- [x] Add hardware-free documentation regressions for scoped content, lazy export
  metadata, and local Markdown file/anchor resolution in
  `particula/tests/dilution_docs_test.py` (P5, #1399).
