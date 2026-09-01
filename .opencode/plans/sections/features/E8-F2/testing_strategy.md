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
- **P3:** Extend `resident_communication_test.py` and communication kernel tests
  for prepared GAS/PARTICLES/volume calls, mode resolution, no-op behavior,
  stable buffers, and absence of P1 scans or readback during enqueue.
- **P4:** Extend condensation kernel and adapter tests with independent launch
  traces and numerical fixtures. Public calls must retain full validation;
  prepared calls must use the same four-substep kernels and supplied sidecars
  without allocation or host-side vapor-pressure work.
- **P5:** Extend coagulation, dilution, wall-loss, and process-adapter tests for
  public-wrapper parity, persistent RNG advancement without reset, selected-box
  reuse, zero-work behavior, and forbidden enqueue-time host operations.
- **P6:** Extend nucleation and exhaustion tests for admission/no-admission,
  activation, resampling precedence, scaling fallback, conservation, fixed
  capacity, failure gating, and fully supplied scratch/status storage.
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
