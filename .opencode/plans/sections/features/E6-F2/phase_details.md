# Phase Details

- [ ] **E6-F2-P1:** Define scalar and per-box dilution input contract with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Freeze coefficient, time, shape, dtype, device, return, and no-op semantics against T1.
  - Files: `particula/gpu/kernels/dilution.py`, `particula/gpu/kernels/tests/dilution_test.py`
  - Tests: Signature, scalar normalization, per-box acceptance, units, and rejection ordering.

- [ ] **E6-F2-P2:** Implement fixed-shape particle and gas dilution kernels with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Apply the T1 finite-step factor in place to particle and gas concentrations only.
  - Files: `particula/gpu/kernels/dilution.py`, `particula/gpu/kernels/__init__.py`
  - Tests: One/multi-box kernels, multi-species gas, inactive/zero concentrations, exact no-ops, field invariants.

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
