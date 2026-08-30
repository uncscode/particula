# Testing Strategy

Every implementation phase ships its own co-located tests. Coverage thresholds
must not be lowered. Test files use the repository's `*_test.py` convention in
`particula/execution/tests/`.

## Per-Phase Coverage

- **P1 — Capability and signature:** Unit-test exact dataclass/enum validation,
  deterministic capability outcomes, complete signature construction, and
  malformed or replaced binding rejection in
  `particula/execution/tests/graph_capture_test.py`. Use fakes for hardware-free
  API-presence tests. Warp CPU must report capture unsupported. Issue #1547
  delivered these cases in `graph_capture_test.py`: hardware-free ordered probe
  outcomes and no-Warp-import subprocess coverage, plus Warp-guarded real
  resident-request signature, representative drift, and stable RNG-array
  identity cases. The declaration test cases do not require CUDA.
- **P2 — Lifecycle and invalidation:** Parametrize every legal and illegal state
  transition, deterministic first invalidation reason, idempotent teardown, and
  read-only rejection behavior. Verify no implicit recapture, launch, transfer,
  synchronization, allocation, reset, or fallback occurs.
- **P3 — Resident recapture integration:** Exercise exact session/registry/guard
  identity, closed-step gates, all structural drift triggers, active-slot
  payload changes that remain compatible, terminal/faulted resident states,
  and writer-may-have-launched fault behavior. Extend
  `particula/execution/tests/full_loop_test.py` only for contract integration;
  CPU/uncaptured/captured numerical parity remains E8-F4 scope.
- **P4 — Documentation:** Validate contract strings or export boundaries where
  existing docs tests apply, and run `mkdocs build --strict` for links and
  publication structure.

## Required Scenarios

- Device/backend/native-device mismatch and absent capture APIs.
- Changes to `n_boxes`, `n_particles`, or `n_species`.
- Replaced session containers, primary arrays, resource views/sidecars,
  diagnostics, graph/schedule, communication map/buffers, thermodynamics or
  process configuration, and RNG sidecars.
- Mutable values with stable identity and shape, including active particle
  counts and advancing RNG words, remain compatible.
- Open guard, inactive/finalized/faulted/closed session, already invalidated
  graph, and recapture attempted without explicit retirement.
- Known unsupported-capture errors are distinguished from validation,
  allocation, cleanup, and launch failures.
- Internal capture names remain absent from `particula.execution` and top-level
  package exports unless a later feature explicitly changes the API boundary.

## Validation Commands and Coverage Evidence

Focused development checks are assertion-only and run with coverage disabled:

```bash
pytest particula/execution/tests/graph_capture_test.py -q --no-cov
pytest particula/execution/tests/full_loop_test.py -q
pytest particula/execution/tests/exports_test.py \
  particula/tests/execution_exports_test.py -q
pytest particula/execution/tests/ -q
```

These focused targets are not coverage evidence. A focused-target coverage run
is invalid evidence, not a feature failure and not a reason to weaken tests.
After focused checks pass, run the untargeted repository runner so the normal
full-package scope and threshold apply:

```bash
.opencode/tools/run_pytest.py
```

If executable resident lifecycle modules change, also retain per-target
term-missing output and the repository's >=80% resident aggregate gate for the
actual changed module list. Run Ruff and mypy through repository policy. CUDA
rows are optional pass-or-clean-skip evidence and must never fall back to CPU.
