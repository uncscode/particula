# Issue 1661 implementation and validation

Branch: `codex/scalable-coagulation-1661`; base
`eea5699f770c4c9a10a132fd7784a1d1ef75f699`. The original checkout is unchanged;
implementation is in a separate worktree. No release/version bump is included.

## Implemented contract

- Removed scientific population ceilings from native resolved coagulation.
  All 15 nonempty additive masks use exact tiled pair reductions with int64
  pair ranks, O(N²) work and bounded reduction scratch. Particle storage is
  linear; actual memory/index limits remain explicit.
- Equal active represented counts are required per physical box. Volumes may
  differ between boxes. No box splitting, weight changes or CPU fallback.
- Before RNG, particle or collision-output writes, admit every box against
  `ceil(M A(A-1)/2 w dt/V) <= min(trial budget, output capacity, floor(A/2))`.
  Reject the entire coagulation operation on excess demand and report the
  rejecting box and resource maximum timestep. There is no silent clipping
  or automatic library subcycling. Earlier/later processes are not rolled back.
- Direct preflight reports demand, capacity, active/pair counts, resource step
  and bound workspace/output bytes. Optional work counters report actual
  scheduled/attempted trials; collision counts report applied acceptance.
- Direct, high-level prepared and low-level resident paths share the sampler.
  Prepared plans/checkpoints freeze resource policy and sampler version.
  Low-level enqueue has no host readback; callers explicitly synchronize to
  observe deferred admission failures. High-level observations fault on failure.
- Capture records without advancing; restart preserves physical/RNG state.
  Last-step work diagnostics reset on preparation/restart. Optional counters
  are ownership-validated and remain unchanged on admission rejection.
- Direct settling uses particle effective density; prepared strategy execution
  retains its documented mean material-density convention.

## Device and reproduction

Python 3.12; Warp 1.11.1. Native CUDA: RTX 5060, 8 GiB, sm_120,
driver 615.78.08, CUDA driver 13.4, Warp toolkit 12.9.
**CUDA is present.** Restricted device access initially hid it; CUDA runs used
explicit access to the installed device, never a CPU fallback.

Use the checkout on `PYTHONPATH` and its selected interpreter. Local runs used
`/home/kyle/Code/particula/.venv/bin/python`. Warp cache was set with
`wp.config.kernel_cache_dir = "/tmp/particula-1661-warp-cache"` before pytest.
Notebook example execution additionally needs local socket access.

Final package command:

```bash
python -m pytest particula -q -Werror -m "not slow and not performance" \
  --cov=particula --cov-report=term --cov-fail-under=80
```

Other gates:

```bash
python -m pytest examples_tests -q -Werror -m "not slow and not performance"
ruff check particula/ examples_tests/ scripts/ --no-fix
ruff format particula/ examples_tests/ scripts/ --check
mypy particula/ --ignore-missing-imports
npm --prefix website test
npm --prefix website run build
python scripts/check_docs_structure.py --site-dir website/dist
```

## Numerical evidence

Independent rate/property/statistical tests cover every additive mask, charged
fits, sparse/empty slots, nonuniform volumes and equal non-unit counts. Exact
serial event/RNG comparisons remain diagnostic oracles, not production fallback.
Budget saturation tests watch particles, outputs, RNG and counters before and
after rejection. Poisoned rate scratch after prior failure verifies that
admission never reads uninitialized indices. Metadata checks cover byte/index
overflow without attempting huge allocations.

Finite-time runs exercised 255/256/257, 2,000, 5,000, 10,000, 16,383/16,384/16,385
and 32,768 active particles on CUDA; additional Warp CPU runs reached 32,768.
Every run checked mass, charge and exact number loss. A zero-event realization
at 257 is valid and is not relabeled as a collision. These sizes are validation
workloads, not supported maxima. Inspection timings include setup/compilation
and do not establish production throughput.

The final bimodal study is `scripts/gpu_studies/coagulation_convergence.py`:

```bash
python scripts/gpu_studies/coagulation_convergence.py --device cuda:0 \
  --ensembles 128 --output docs/data/gpu/coagulation-convergence-1661.json
```

At 32/64/128 particles, compare three step refinements with independent CPU
continuous-time Gillespie trajectories using CPU strategy matrices. Both paths
resolve a 7:1 small/collector ratio at fixed number density. Number, small-mode
and surface-area fractions at the finest step agree within 1.78 combined
standard errors (predeclared gate: 5 errors plus one ensemble event). Separate
mass/charge and count-loss assertions pass. Saved data and a visually inspected
figure accompany the scientific-correctness validation page. This does not
establish formal convergence order, monotonic stochastic errors, or adequate
sampling at a 10⁶:1 number ratio.

## Examples and integration

The new scalable example shows explicit preflight, resource-admitted steps,
executed counters and conservation. Its complete Warp CPU workload was refreshed
and its figure inspected. The existing direct coagulation example now chooses
explicit admitted steps; its complete CUDA workload and the unchanged
five-process CUDA workload were refreshed and figures inspected.

Oversized historical test steps had depended on silently clipped demand.
Updated fixtures derive admitted steps from independent CPU rates and retain
mandatory events, per-species inventory, untouched fields, ownership and exact
restart assertions. The coupled source fixture still delivers two particles
and checks 1 s checkpoint/3 s continuation; it explicitly delivers the pulse
in the first 0.01 s step before disabling its precursor.

## Validation outcomes

| Gate | Final outcome |
| --- | --- |
| Full package with native CUDA and coverage | 11,046 passed, 26 skipped, 65 deselected, one expected failure; 460.14 s |
| Coverage | 90.66%, exceeding the 80% gate |
| Mypy | 698 source files, no issues |
| Repository Ruff | All checks passed; 783 files already formatted |
| Website tests | 62 Python tests and 14 Node tests passed |
| Site build | 573 pages built; verifier: 703 HTML files, 333 API modules, 1,860 legacy anchors, 130 redirects, zero errors |
| Rendered structure/topic checks | Passed |
| Bimodal CPU-reference/CUDA study | All three population sizes and refinement checks passed |
| CUDA boundary runs | All ten workloads through 32,768 particles passed |

Package log: `/tmp/particula-1661-full-final.log`. The final package run includes
new counters, native capture/restart, and the corrected cloud fixtures. Earlier
oversized-step fixture failures and a test-call seed typo were resolved before
this successful full run. Notebook tests initially failed under restricted
local-socket access and passed with that access restored. No scientific
assertions or timeouts were relaxed.

The final CUDA boundary dataset is
`docs/data/gpu/coagulation-scaling-1661.json`; it includes actual work counters
and current bytes. The 32,768-particle case uses 1,998,948 bound workspace/output
bytes, excluding particles. Timings explicitly include setup and concurrent
validation; no throughput claim is made.

Changed-example execution: **60 passed, 30 policy skips, zero failures**.
CUDA/long examples require explicit refresh; the affected CUDA examples were
refreshed separately in full. Log: `/tmp/particula-1661-doc-examples-final.log`.

Final example regressions: **242 passed, one skipped, zero failures** in
369.87 s with native CUDA access. Log: `/tmp/particula-1661-examples-final.log`.
All requested implementation, examples and validation are complete.

## Upstream integration

Merged upstream `57317f47c1d66d1109f3174484cfd6247b9e7f16` after opening
PR #1663. Upstream changed documentation topics only; no scientific library
code changed. The sole conflict was resolved by retaining the new hierarchy
and placing the scalable example under `gpu-direct`. Post-merge website tests
passed **66 Python + 15 Node tests**. The site rebuilt **597 pages** and passed
link/API/redirect verification with zero errors. The graph validates with
57 topics, 171 resources and 35 supporting pages.

Issue #1679 adds opt-in adaptive outer intervals on top of the fixed sampler
qualified here. The follow-up records evolving-state admission, rollback,
exact-device retry and two 600 s CUDA runs.
[Adaptive validation](../docs/studies/validation/coagulation/direct-coagulation/qualification.md#adaptive-outer-intervals-on-native-cuda)
