# Phase Details

## Sequencing

E6-F5 is required. Complete P1 through P5 in order before P6 conservation
validation; P7 documents only the validated policy, precedence, and bounds.

- [ ] **E6-F6-P1:** Freeze exhaustion policy, precedence, and conservation contracts with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Define defaults, independent controls, transactional failure ordering, diagnostics, and exact versus bounded invariants.
  - Files: `particula/particles/exhaustion.py`, `particula/particles/tests/exhaustion_test.py`
  - Tests: Config defaults, policy truth table, capacity/no-op paths, invalid controls, weighted-inventory oracle, and no-write snapshots.

- [ ] **E6-F6-P2:** Implement deterministic CPU resampling reference with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Precompute and apply deterministic equal-weight conservative remapping that frees enough slots while preserving required moments.
  - Files: `particula/particles/exhaustion.py`, `particula/particles/tests/exhaustion_test.py`, `particula/particles/__init__.py`
  - Tests: Sparse/full boxes, stable ordering/tie breaks, multi-species mass/number/charge conservation, free-slot clearing, and bounded distribution distortion.

- [ ] **E6-F6-P3:** Implement allocation-stable Warp resampling with parity tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Port the CPU plan/commit algorithm using caller-owned active-device sidecars and stable launches.
  - Files: `particula/gpu/kernels/exhaustion.py`, `particula/gpu/kernels/tests/exhaustion_test.py`, `particula/gpu/kernels/__init__.py`
  - Tests: Warp CPU deterministic parity, supplied-buffer identity, shape/dtype/device validation, exact diagnostics, optional CUDA, and failure atomicity.

- [ ] **E6-F6-P4:** Add optional CPU and Warp representative-volume scaling with tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Scale per-box representative volume, raw weights, and source demand in the same direction under explicit bounds while preserving intensive concentrations.
  - Files: `particula/particles/exhaustion.py`, `particula/gpu/kernels/exhaustion.py`, corresponding `*_test.py` files
  - Tests: CPU/Warp scale-factor parity, per-box isolation, exact same-direction updates, source-demand transformation, identity preservation, bound failures, and concentration conservation.

- [ ] **E6-F6-P5:** Enforce independent controls, resampling-first precedence, and fail-closed validation
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Compose E6-F5 discovery with exhaustion planning so resampling is attempted first and scaling is only a configured fallback.
  - Files: CPU/GPU exhaustion and slot-management modules plus their tests
  - Tests: Four control combinations, resampling-sufficient and fallback cases, both-off pre-mutation error, unsatisfiable request, diagnostics, and no silent truncation.

- [ ] **E6-F6-P6:** Validate sparse, full, and over-capacity conservation across CPU and Warp
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Prove cross-backend behavior with an independent NumPy oracle and downstream-shaped source requests.
  - Files: `particula/particles/tests/exhaustion_test.py`, `particula/gpu/kernels/tests/exhaustion_test.py`, integration fixtures
  - Tests: Multi-box/species matrix, physical inventory before/after plus admitted demand, moment/error bounds, repeated calls, exact no-ops, and CUDA clean skips.

- [ ] **E6-F6-P7:** Update development documentation for slot exhaustion policies
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Publish defaults, precedence, equations, diagnostics, direct imports, failure boundaries, dependencies, and deferred capabilities.
  - Files: `AGENTS.md`, `docs/Features/`, `docs/Theory/Technical/Dynamics/Nucleation_Equations.md`, E6 plan sections
  - Tests: Markdown links, API snippets, shape/equation review, terminology, and focused commands.
