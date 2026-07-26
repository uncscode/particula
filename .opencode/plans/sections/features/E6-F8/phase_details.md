# Phase Details

## Sequencing

E6-F5, E6-F6, and E6-F7 are required. Complete P1/P2 before P3, P4, and P5;
run P6 parity and conservation checks before P7 documents the direct step.

- [x] **E6-F8-P1:** Define direct GPU nucleation configuration and preflight with unit tests
  - Issue: #1438 | Size: S | Status: Complete (2026-07-25)
  - Delivered: frozen configuration and sidecar dataclasses plus private,
    read-only Warp preflight matching the P1 ownership and failure-before-write
    boundary. It has no export, direct step, rate computation, mutation,
    hidden transfer, or fallback allocation.
  - Files: `particula/gpu/kernels/nucleation.py`, `particula/gpu/kernels/tests/nucleation_test.py`
  - Tests: Shape/dtype/device/alias validation, scientific-domain rejection, exact no-ops, and snapshots proving no state or sidecar mutation.

- [x] **E6-F8-P2:** Implement device nucleation rate and gas admission with unit tests
  - Issue: #1439 | Size: S | Status: Complete (2026-07-25)
  - Delivered: private `_plan_nucleation_demand(...)` reuses P1 preflight to
    compute survival-included activation/kinetic `J`, `E_pot = J * dt`, and a
    shared per-box inventory-limited accepted demand. Its single commit writes
    only P2 demand/removal/gate sidecars, preserving P3 request buffers,
    particles, and gas concentration.
  - Files: `particula/gpu/kernels/nucleation.py`, `particula/gpu/kernels/tests/nucleation_test.py`
  - Tests: Activation/kinetic oracle comparisons; limiting species and
    lowest-index ties; gate precedence; multi-box admission; nonparticipant
    zero removal; ULP-safe inventory correction; zero-capacity/empty-box cases;
    and identity/immutability snapshots.

- [x] **E6-F8-P3:** Integrate fixed-shape slot activation sidecars with unit tests
  - Issue: #1440 | Size: S | Status: Complete (2026-07-25)
  - Delivered: private `_stage_nucleation_slots(...)` converts exact finite
    nonnegative P2 demand-volume products to full int32 provisional counts,
    reuses E6-F5 diagnostics, and writes only caller-owned P3 sidecars. It
    retains over-capacity counts and writes the bounded free-slot prefix with
    `-1` tails; it neither invokes E6-F6 nor activates slots or mutates
    particles/gas.
  - Files: `particula/gpu/kernels/nucleation.py`, `particula/gpu/kernels/tests/nucleation_test.py`, `.opencode/guides/architecture/architecture_outline.md`, `.opencode/guides/architecture_reference.md`
  - Tests: Normal/capacity-boundary/empty layouts; exact integer conversion and
    int32 upper bound; malformed, alias, conversion, and E6-F5 failures with
    snapshots; ownership, identity, and Warp CPU/optional-CUDA coverage.

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
