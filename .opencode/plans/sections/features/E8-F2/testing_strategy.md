# Testing Strategy

Every production phase ships co-located `*_test.py` coverage in the same PR.
Coverage thresholds and the normal collection policy must not be lowered.

## Per-Phase Approach

- **P1 (implemented, #1552):** `resident_enqueue_test.py` covers exact frozen
  prepared carriers; READY E8-F1 attachment/lifecycle/signature gates; canonical
  schedule, duration, and structural-drift rejection; and read-only rejection.
   It traps guard-token entry during successful preparation. Other forbidden
   host-operation traps remain coverage for their owning later phases.
   `diagnostics_test.py`, `resident_communication_test.py`,
  `graph_capture_test.py`, `full_loop_test.py`, and `exports_test.py` provide
  adjacent regression evidence for shared validators, unchanged scheduler
  behavior, and concrete-only exports.
- **P2 (implemented, #1553):** `state_updates_test.py`,
  `thermodynamic_updates_test.py`, and `diagnostics_test.py` cover P1-bound
  setup identity/pinning validation and validate-once/enqueue-only boundaries.
  They assert state and diagnostics writer order, thermodynamic cursor/freshness
  behavior, valid empty no-ops, and standalone-path compatibility. Forbidden
  operation traps cover allocation, host readback/synchronization, binding or
  registry validation, and schedule/registration/freshness resolution after
  successful preparation.
- **P3 (implemented, #1554):** `resident_communication_test.py` and
  `communication_test.py` cover P1-bound closed GAS/PARTICLES setup, exact
  resource/primary/work-array retention, structural and identity-drift
  rejection, and prepared native dispatch. They exercise absent, equal, and
  changed final-volume sidecars; equal-volume write-free behavior; empty/zero
  barriers; communication-before-volume ordering; and GAS aggregate-overdraw
  commit gating. Scoped spies prove enqueue does not revalidate, resolve maps,
  reacquire resources, allocate, transfer/read back, or synchronize.
- **P4 (implemented, #1555):** `condensation_test.py` and
  `condensation_adapter_test.py` cover the private prepared kernel call and
  concrete adapter binding. They retain direct-wrapper validation, fallback and
  supplied-sidecar identity coverage, the four equal gas-coupled
  inventory-limited substeps, and adapter binding behavior. Focused
  instrumentation verifies prepared enqueue performs no validation, allocation,
  host refresh/readback, synchronization, or resource lookup. Local docstrings
  document the setup/enqueue ownership and failure boundary; no user-facing
  documentation validation was required.
- **P5 (implemented, #1556):** focused coagulation, dilution, wall-loss, and
  resident-adapter tests cover prepared/direct-wrapper delegation, frozen
  references, stable returned and sidecar identities, dilution no-ops,
  persistent Brownian RNG advancement without reset, and wall-loss all/partial/
  empty selected lanes. Enqueue-only traps verify that prepared dispatch does
  not repeat setup work or re-enter public wrappers.
- **P6 (implemented, #1557):** `nucleation_test.py`, `exhaustion_test.py`,
  `nucleation_parity_test.py`, and `process_adapters_test.py` cover complete
  preparation validation/preallocation, direct-wrapper versus prepared behavior,
  live payload scans, pinned-array rebinding, dynamic writer gating, fixed-policy
  non-substitution, sidecar and return identities, resampling precedence,
  scaling fallback, no-admission primary no-ops, exact error precedence, and
  tight particle-plus-gas conservation. Scoped traps prove enqueue performs no
  allocation, readback, synchronization, mutable-carrier lookup, or legacy/public
  executor call. Resident binding remains enqueue-only and injected public-
  resolver fallback coverage remains intact.
- **P7:** Add full resident integration coverage for canonical operation order,
  uncaptured prepared output regression, one-token lifecycle behavior, and
  writer-fault handling. CUDA graph capture is optional pass-or-clean-skip
  evidence; Warp CPU validates the uncaptured contract and must never masquerade
  as captured execution.
- **P8:** Run documentation contract tests and `mkdocs build --strict`.

## Validation and Coverage Policy

Focused development checks are assertion-only and coverage disabled, for
example:

```bash
pytest particula/execution/tests/resident_enqueue_test.py -q
pytest particula/execution/tests/ -q
pytest particula/gpu/kernels/tests/ -q -k "prepared or resident"
```

A focused target cannot supply repository coverage evidence. If a focused run
passes assertions but cannot satisfy a package-wide coverage threshold, treat
that as invalid coverage evidence, not an implementation or fix failure. After
focused checks pass, run the full applicable suite through the untargeted
repository runner:

```bash
.opencode/tools/run_pytest.py
```

That command supplies repository-configured full-package coverage and the
normal threshold. Do not pass focused targets or ad hoc `--cov` overrides to
it. Also run changed-module resident coverage when required by the testing
guide, preserving per-target term-missing rows and the aggregate 80% gate.

Use explicit float64 fixtures and independent references for deterministic
physics. Keep conservation assertions separate and tight. Stochastic tests use
aggregate bounds, not exact CPU/CUDA seed replay. CUDA capture rows use the
registered `warp` and `cuda` markers and skip cleanly when no qualified device
exists; there is no CPU fallback.
