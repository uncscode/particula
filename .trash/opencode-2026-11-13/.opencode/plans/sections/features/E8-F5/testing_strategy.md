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
- **P3 (shipped, #1577):** `captured_full_loop_test.py` normalizes Warp device
  objects to opaque native strings, retains only CUDA candidates, and marks
  native rows `warp`, `cuda`, and
  `gpu_parity`, and skips before capture when CUDA, the capture API, or a
  candidate is unavailable. Only explicit runtime, device, or capture-API
  unavailability may skip; binding, lifecycle, and resource failures propagate.
  It captures qualified CUDA GAS and PARTICLES
  closed-map bindings for active, prescribed-volume, and no-work scenarios and
  compares replay snapshots (including diagnostics and family buffers) with an
  independently enqueued Warp-CPU binding. Scoped replay instrumentation rejects
  visible conversion, allocation, copy, module and array-method readback,
  synchronization, and registry
  resource work; a hermetic rejection row proves qualification fails before
  capture or guard entry. No CPU or Warp-CPU capture fallback is used.
- **P4 (shipped, #1578):** `rng_invariance_test.py` validates independent real
  Brownian/wall-loss stream advancement, selected-lane reset, and no hidden
  normal-dispatch reset. `checkpoint_test.py` validates schema-v4 same-device
  continuation of both advanced streams without stream initialization.
  `graph_capture_test.py` validates forged, attachment, signature, terminal,
  and lifecycle replay rejections before launch plus launch- and completion-
  failure fault/revocation/release cases. `captured_full_loop_test.py` adds
  optional native-CUDA aggregate evidence that independently isolated wall-loss
  and coagulation streams advance and have non-vacuous activity.
- **P5 (shipped, #1579):** The dated integrated closeout below records the
  focused matrix, untargeted coverage runner, documentation contract tests, and
  strict MkDocs validation. It adds documentation evidence only; it does not
  change production behavior or P1--P4 fixtures.

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
  particula/execution/tests/checkpoint_test.py -q --no-cov
pytest particula/execution/tests/captured_full_loop_test.py -q \
  -m "warp and cuda" --no-cov
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

## E8-F5-P5 authoritative integrated validation evidence

**Evidence record date:** 2026-09-05. This record follows the P4 plan update
dated 2026-09-05 and preserves the original literal command outcomes. **Runtime
preflight:** Python 3.12.12; Warp was installed because the required uncaptured
matrix executed with 48 passes. The available test wrapper did not report a
Warp version or enumerate opaque device strings. No native CUDA device/capture
prerequisite was qualified: the optional selector cleanly skipped all 11 CUDA
rows. This is not CPU or Warp-CPU capture fallback.

Required focused rows use `--no-cov`; the wrapper enforced equivalent disabled
coverage. The untargeted runner is the sole full-package coverage evidence.

| Kind | Exact command | Exit status | Literal outcome |
| --- | --- | --- | --- |
| Focused required | `pytest particula/execution/tests/captured_full_loop_test.py -q --no-cov` | 0 | `48 passed, 11 skipped` |
| Focused required | `pytest particula/execution/tests/graph_capture_test.py particula/execution/tests/rng_invariance_test.py particula/execution/tests/checkpoint_test.py -q --no-cov` | 0 | `274 passed, 1 skipped` |
| Focused optional CUDA | `pytest particula/execution/tests/captured_full_loop_test.py -q -m "warp and cuda" --no-cov` | 0 | `11 skipped, 48 deselected` (clean skip) |
| Untargeted coverage | `.opencode/tools/run_pytest.py` | 0 | `6634 passed, 24 skipped, 1 xfailed, 92.92% coverage` |
| Documentation required | `pytest particula/execution/tests/graph_capture_docs_test.py particula/tests/execution_selection_docs_test.py -q --no-cov` | 0 | `25 passed` |
| Documentation required | `mkdocs build --strict` | 0 | Passed through the approved `docs-validator` `build_mkdocs_validate` worktree wrapper; strict mode is intrinsic and the exact workflow worktree is supplied as `cwd`. |

The required installed-Warp uncaptured evidence passed. Deterministic fields use
`rtol=1e-12, atol=1e-30`; independent concentration-weighted per-box/per-species
conservation remains separate; stochastic evidence uses aggregate or sigma-bounded
criteria rather than exact per-seed or cross-device RNG-word replay. The optional
native-CUDA row does not alter that mandatory gate. The approved
`docs-validator` `build_mkdocs_validate` strict worktree validation passed after
the original evidence record, so P5 is shipped.
