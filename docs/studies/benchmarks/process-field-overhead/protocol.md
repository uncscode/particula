# Custom processes and fields: qualification

This development record covers the `codex/process-field-extensions` branch.
Local qualification is complete; this record does not imply merge or release.

## Performance protocol

The comparison baseline is pre-feature revision
`a73fa19741885844ffeebbf76723a8b60ab39223`. Extract it with `git archive` and
use the same interpreter, CUDA device and benchmark script for both revisions.
The script is `scripts/gpu_studies/extension_overhead.py`.

Before collecting results, the acceptance bound for the extension-free workload
is set to **20% median overhead** relative to the baseline, separately for
uncaptured and captured execution. This is a bounded regression check for a
small launch-sensitive workload, not a guarantee for all scientific simulations.
If the bound fails, investigate the cause and report it; do not change the bound
to fit the observed measurements.

Use eight boxes with 256 particle slots, native dilution at 0.01/s, a 0.001 s
timestep, 20 warm-up steps and seven samples of 2,000 steps. Synchronize after
warm-up and each timed batch; exclude compilation and preparation. Run baseline
and current revisions serially without competing GPU tests, and repeat in
reverse order if the result is near the bound or has substantial variability.
The report records every sample, source hashes, revision and runtime versions.

Also measure the published prescribed-warming custom process composed with
native dilution in resident, captured resident and explicit CPU-transfer modes.
Use 200 steps per sample for transfers, whose cost includes downloads, staged
Python callbacks, validation and uploads. Report absolute time per step; these
different execution paths do not establish a general GPU speedup.

Every timing run checks completed steps and the analytical native gas decay.
Custom runs additionally check the integrated time-dependent warming field.
Numerical, failure, ownership, capture and restart tests remain separate gates.

## Correctness and publication checks

Library revision `8d108f454`, Python 3.12.12, real NVIDIA GeForce RTX 5060:

| Check | Result |
| --- | --- |
| Full package, warnings as errors, no slow/performance exclusion | 10,312 passed, 20 skipped, one expected failure |
| Full-package coverage | 91.79%; required minimum 80% |
| CI example suite | 238 passed, one skipped |
| Full script suite | 294 passed |
| Whole-library mypy | Passed, 612 source files |
| CI Ruff and formatting targets | Passed, 685 files formatted |
| Website tests | 38 Python and 11 JavaScript tests passed |
| Three new examples | Full CPU/CUDA refresh and saved-data replot passed |

The package run uses a separate coverage database so other local runs cannot
overwrite its evidence:

```bash
COVERAGE_FILE=.artifacts/process-field-extensions/.coverage-identity \
WARP_CACHE_PATH=/tmp/particula-extension-warp \
MPLCONFIGDIR=/tmp/particula-mpl \
.venv/bin/pytest -q -Werror --cov=particula --cov-report=term-missing \
  --cov-report=json:.artifacts/process-field-extensions/coverage-identity.json \
  --cov-fail-under=80
.venv/bin/pytest examples_tests -q -Werror \
  -m 'not slow and not performance' --no-cov
.venv/bin/pytest scripts/tests -q -Werror --no-cov
```

Focused tests cover CPU later-box rollback and managed RNG; field copy/view
isolation and source/removal accounting; real CUDA resident and explicit-transfer
execution; device pointer retention; failure suppression; checkpoint drift
rejection; and capture recording/replay. Declared saturation consumers also match an independent ideal-gas oracle after
native updates, including transfer, resident, capture and restart paths.
Grouped sequences also preserve absolute time across calls and prepared
restart; delimiter-bearing extension identities retain independent CPU RNG
streams, including explicit CPU transfers.
Native numerical regression remains
part of the full package suite. The new downloadable GPU example also executes
from its extracted ZIP outside the checkout, including its declared Warp helper.

The first qualification run found five outdated private test spies, which did
not accept the field-store keyword. They now retain the original identity,
rollback and numerical checks while accepting the current private contract.
The restricted website test stalled during notebook-kernel startup; its host
rerun passed. Neither run was counted as a successful final package gate.

## Timing results

[Read the measured timings and their scope](../process-field-overhead.md#timing-results).

For each source tree, reproduce the native case with:

```bash
.venv/bin/python scripts/gpu_studies/extension_overhead.py \
  --source-root /path/to/source --revision SOURCE_REVISION \
  --mode native --steps 2000 --repeats 7 --output result.json
```

Add `--capture` for captured cases. Use the current source tree and
`--mode resident` for the custom resident workload, or `--mode transfer
--steps 200` for explicit host execution. Baseline and current runs must use the
same interpreter and script. CUDA is mandatory; the harness rejects CPU fallback.

## Delivery qualification

Saved-example validation passed, with CUDA workloads explicitly skipped by that
gate and the new CUDA example separately refreshed and tested on the GPU.
The final site rebuild passed with zero rendered-link errors across 607 HTML
pages and 286 API modules. Hosted PR #1652 checks passed on `662a8119a`,
including the Python 3.12/3.13/3.14 matrix, documentation, lint, packaging,
conda and plugin validation. The live saturation review finding is resolved.
Final-head status remains visible on the PR; this record does not authorize
merge or release.
