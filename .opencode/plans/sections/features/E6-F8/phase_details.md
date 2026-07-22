# Phase Details

- [ ] **E6-F8-P1:** Define direct GPU nucleation configuration and preflight with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Freeze device configuration, fixed-shape sidecars, validation order, ownership, and failure-before-write contract matching E6-F7.
  - Files: `particula/gpu/kernels/nucleation.py`, `particula/gpu/kernels/tests/nucleation_test.py`
  - Tests: Shape/dtype/device/alias validation, scientific-domain rejection, exact no-ops, and snapshots proving no state or sidecar mutation.

- [ ] **E6-F8-P2:** Implement device nucleation rate and gas admission with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Compute potential events and one shared gas-feasible admission factor per box before any source or gas write.
  - Files: `particula/gpu/kernels/nucleation.py`, `particula/gpu/kernels/tests/nucleation_test.py`
  - Tests: Activation/kinetic parity, each limiting species, zero inventory/rate/time, nonnegative gas, and provisional-demand diagnostics.

- [ ] **E6-F8-P3:** Build provisional fixed-shape slot requests with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Package provisional gas-admitted demand into E6-F5 request/count sidecars without shape or identity changes.
  - Files: `particula/gpu/kernels/nucleation.py`, E6-F5 slot module, GPU kernel tests
  - Tests: Empty/sparse/exact-capacity boxes, deterministic indices, `-1` tails, exact counts, preserved unselected fields, and no duplicate slot model.

- [ ] **E6-F8-P4:** Integrate device exhaustion planning without fallback with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Consume E6-F6 resampling-first/scaling plans, finalize scaled demand and requests, and reject unsatisfied plans before writes.
  - Files: `particula/gpu/kernels/nucleation.py`, E6-F6 exhaustion module, GPU kernel tests
  - Tests: Full slots, policy combinations, precedence, insufficient scratch, scaled-demand diagnostics, unsatisfiable demand, no final-domain residual, and conservation snapshots.

- [ ] **E6-F8-P5:** Add atomic direct GPU nucleation step with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Expose one low-level step that preflights all boxes, plans capacity, then commits matching particle activation and gas depletion on device.
  - Files: `particula/gpu/kernels/nucleation.py`, `particula/gpu/kernels/__init__.py`, GPU kernel tests
  - Tests: Return and supplied-buffer identity, repeated calls, all-box atomicity, explicit inputs, no fallback/transfer, and mutation boundaries.

- [ ] **E6-F8-P6:** Validate CPU parity and per-species conservation with integration tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Compare the direct Warp step with an independent E6-F7 float64 oracle over representative multi-box/multi-species capacity cases.
  - Files: `particula/gpu/kernels/tests/nucleation_parity_test.py`, test support fixtures
  - Tests: Warp CPU required, optional CUDA, rate/admission parity, repeated calls, and per-box/species particle-plus-gas conservation.

- [ ] **E6-F8-P7:** Update development documentation for direct GPU nucleation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Publish the bounded API, ownership, transfer, sidecar, conservation, dependency, and no-fallback contracts.
  - Files: `AGENTS.md`, `docs/Features/`, `docs/Theory/Technical/Dynamics/Nucleation_Equations.md`, `docs/Examples/Nucleation/`, E6 sections
  - Tests: Link/import validation, equation review, focused commands, and explicit-transfer example execution where applicable.
