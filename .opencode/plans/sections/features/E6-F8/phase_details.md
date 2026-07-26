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

- [x] **E6-F8-P4:** Integrate private device exhaustion policy without fallback with unit tests
  - Issue: #1441 | Size: S | Status: Complete (2026-07-25)
  - Delivered: private `_orchestrate_nucleation_exhaustion(...)` consumes
    immutable P2/P3 handoffs, chooses fully viable resampling first and scaling
    fallback second, and writes P4 final demand/count/free-prefix diagnostics.
    Expected all-box rejections precede P4 writes and primitive entry; entered
    primitive failures retain their documented no-cross-primitive-rollback
    boundary. No activation, particle/gas mutation, public API, or E6-F9
    integration was added.
  - Files: `particula/gpu/kernels/nucleation.py`,
    `particula/gpu/kernels/tests/nucleation_test.py`, architecture and feature
    documentation.
  - Tests: Independent policy-oracle, precedence/fallback/mixed-box and boundary
    cases; final diagnostics/identities; complete expected-rejection snapshots;
    and separate entered-primitive boundary coverage on Warp CPU with optional
    CUDA skips.

- [x] **E6-F8-P5:** Add atomic direct GPU nucleation step with unit tests
  - Issue: #1442 | Size: S | Status: Complete (2026-07-25)
  - Delivered: supported lazily exported `nucleation_step_gpu(...)` composes
    P1--P4 and performs P5 handoff validation followed by one fused device
    commit. It initializes only finalized selected free slots and subtracts the
    matching finalized gas mass, returning the identical containers.
  - Files: `particula/gpu/kernels/nucleation.py`,
    `particula/gpu/kernels/__init__.py`,
    `particula/gpu/kernels/tests/nucleation_test.py`, and
    `particula/gpu/tests/kernel_exports_test.py`.
  - Tests: nominal/multi-box commits, explicit and environment inputs, repeated
    current-gas calls, P4 resampling/scaling integration, no-work paths,
    malformed/rebound P5 handoffs, precommit atomicity, caller-sidecar identity,
    and lazy-export/no-hidden-transfer guards.

- [x] **E6-F8-P6:** Validate CPU parity and per-species conservation with integration tests
  - Issue: #1443 | Size: S | Status: Complete (2026-07-26)
  - Delivered: 718-line independent NumPy float64 direct-Warp parity and
    conservation suite. This is coverage-only and does not change the public API
    or runtime behavior.
  - Files: `particula/gpu/kernels/tests/nucleation_parity_test.py`
  - Tests: Warp CPU with optional clean CUDA skips; activation/kinetic P2/P3/P5
    parity; exact write-free gates; per-box/species matrix inventory; separate
    scaling inventory accounting; resampling precedence; repeated current-gas
    calls; preflight non-mutation; and zero-box/zero-capacity boundaries.

- [ ] **E6-F8-P7:** Update development documentation for direct GPU nucleation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Publish the bounded API, ownership, transfer, sidecar, conservation, dependency, and no-fallback contracts.
  - Files: `AGENTS.md`, `docs/Features/`, `docs/Theory/Technical/Dynamics/Nucleation_Equations.md`, `docs/Examples/Nucleation/`, E6 sections
  - Tests: Link/import validation, equation review, focused commands, and explicit-transfer example execution where applicable.

## P7: Documentation and example — Shipped (#1444)

Published the direct-Warp P1--P5 boundary, caller-owned sidecars, failure
limits, Warp CPU baseline, optional CUDA skip expectations, and explicit
transfer/synchronization example. Focused inventory tolerance is
`rtol=1e-12, atol=1e-30`.
