# Testing Strategy

Every phase ships with co-located `*_test.py` coverage. No coverage threshold is
lowered, and production behavior is not changed solely to expose test internals.

## Per-Phase Approach

- **P1 (shipped, #1575):** CPU-only unit tests in
  `captured_full_loop_test.py` validate an immutable two-box/multi-species
  scenario and an independent NumPy oracle. They cover closed gas communication,
  volume evolution, dilution, saturation, six diagnostics, per-box/species
  inventory, no-work/inactive-slot behavior, and invalid input rejection.
- **P2 (shipped, #1576):** `captured_full_loop_test.py` runs the real READY
  prepared uncaptured path on Warp CPU over multiple timesteps. It compares
  fields, closed-GAS work buffers, accounting, and six diagnostics individually
  with the P1 oracle; separately checks conservation and stable identities; and
  uses scoped spies to reject enqueue-time setup, allocation, upload, readback,
  and synchronization. A zero-duration row verifies write-free preservation.
- **P3 (shipped, #1577):** `captured_full_loop_test.py` discovers only opaque
  CUDA string candidates from Warp, marks native rows `warp`, `cuda`, and
  `gpu_parity`, and skips before capture when CUDA, the capture API, or a
  candidate is unavailable. It captures qualified CUDA GAS and PARTICLES
  closed-map bindings for active, prescribed-volume, and no-work scenarios and
  compares replay snapshots (including diagnostics and family buffers) with an
  independently enqueued Warp-CPU binding. Scoped replay instrumentation rejects
  visible conversion, allocation, copy, readback, synchronization, and registry
  resource work; a hermetic rejection row proves qualification fails before
  capture or guard entry. No CPU or Warp-CPU capture fallback is used.
- **P4:** Use deterministic lifecycle tables plus `stochastic` aggregate rows for
  RNG. Assert explicit reset and continuation semantics separately from exact
  state parity; verify no graph launch after preflight rejection.
- **P5:** Run the integrated focused matrix, documentation contract tests, and
  strict MkDocs validation.

## Numerical Policy

- Use explicit `np.float64` fixtures and independent expected calculations.
- Deterministic parity uses explicit per-field `rtol`/`atol`; initial targets are
  `1e-12` and `1e-30`, with any exception justified beside the assertion.
- Conservation is a separate per-box/per-species concentration-weighted check
  with tight bounds; a stochastic tolerance may not relax conservation.
- Stochastic outcomes use documented aggregate or sigma bounds, not exact
  CPU/Warp/CUDA seed-by-seed replay.

## Commands and Coverage

Focused fix checks are assertion-only and coverage disabled:

```bash
pytest particula/execution/tests/captured_full_loop_test.py -q --no-cov
pytest particula/execution/tests/graph_capture_test.py \
  particula/execution/tests/rng_invariance_test.py \
  particula/execution/tests/checkpoint_test.py -q
pytest particula/execution/tests/captured_full_loop_test.py -q \
  -m "warp and cuda"
```

A focused target with `--cov` is invalid comprehensive evidence; inability to
meet full-package coverage from a focused file is a validation-infrastructure
mistake, not a feature failure. After focused checks pass, run the repository's
untargeted suite, which supplies configured full-package coverage and its normal
threshold:

```bash
.opencode/tools/run_pytest.py
mkdocs build --strict
```

If a required command is unavailable, record it as unavailable rather than
inferring pass. Optional CUDA rows may pass or cleanly skip.
