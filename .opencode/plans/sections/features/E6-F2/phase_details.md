# Phase Details

## Sequencing

E6-F1 must provide the canonical CPU contract before P1. Complete P1 through
P4 in order; P5 documents only the shipped implementation and parity evidence.

- [x] **E6-F2-P1:** Define scalar and per-box dilution input contract with unit tests
  - Issue: #1395 | Size: S | Status: Completed
  - Delivered: `dilution_step_gpu` is concrete-module-only and validates finite
    nonnegative scalar coefficient/time inputs plus same-device `wp.float64`
    `(n_boxes,)` coefficient metadata. Valid calls return identical containers
    with no launch or caller-state write; per-box values and container state are
    deferred to P3, and package re-export is deferred to P2.
  - Files: `particula/gpu/kernels/dilution.py`, `particula/gpu/kernels/tests/dilution_test.py`
  - Tests: Signature/import boundary, scalar normalization, per-box identity,
    zero-box/no-write paths, units, and rejection ordering.

- [x] **E6-F2-P2:** Implement fixed-shape particle and gas dilution kernels with unit tests
  - Issue: #1396 | Size: S | Status: Completed
  - Delivered: `dilution_step_gpu` applies `exp(-alpha * time_step)` in place
    to fixed-shape particle and gas concentrations, returns the original
    containers, preserves protected fields, and retains scalar/per-box
    coefficient and zero-time/scalar-zero no-op paths. Only this entry point is
    exported through `particula.gpu.kernels`.
  - Files: `particula/gpu/kernels/dilution.py`, `particula/gpu/kernels/__init__.py`, `particula/gpu/kernels/tests/dilution_test.py`
  - Boundary: Per-box coefficient-value validation, complete preflight, and
    rollback remain P3 work.

- [ ] **E6-F2-P3:** Add atomic entry-point preflight validation with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Reject invalid physical values and structural metadata before launches, allocation, or mutation.
  - Files: `particula/gpu/kernels/dilution.py`, `particula/gpu/kernels/tests/dilution_test.py`
  - Tests: Invalid type/rank/shape/dtype/device/domain/state and snapshot-based atomicity.

- [ ] **E6-F2-P4:** Add CPU and Warp multi-box parity and invariant tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Record float64 tolerances and prove deterministic parity with E6-F1 across the acceptance matrix.
  - Files: `particula/gpu/kernels/tests/dilution_test.py`
  - Tests: Warp CPU required; optional CUDA; scalar/per-box, particle/gas, repeated-step, identity, and protected-field cases.

- [ ] **E6-F2-P5:** Update development documentation for direct GPU dilution
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document direct imports, explicit transfers, supported inputs, validation, parity evidence, and deferred scope.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`, `docs/Features/Roadmap/data-oriented-gpu.md`, `AGENTS.md`
  - Tests: Documentation links/import snippets and focused test command verification.
