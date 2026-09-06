# Testing Strategy

Every phase ships co-located `*_test.py` coverage. Default tests validate
records, parsing, timing control flow, analysis, and documentation without CUDA
or NVIDIA tools. Real timings and profiles remain opt-in evidence, never stable
assertion thresholds.

## Per-Phase Approach

- **P1 (delivered):** `particula/gpu/tests/profiling_support_test.py` covers
  strict records, canonical ordering and JSON round-trips, workload IDs,
  machine metadata allow-lists, safe injected-root paths and streaming
  checksums, invalid dimensions/units, and explicit unavailable records. These
  tests are hardware-free and do not import Warp or invoke CUDA/profiler tools.
- **P2 (delivered):** Hardware-free timer, operation, synchronizer, reset, raw
  report, provenance, serializer, and replacement spies prove host-launch has
  only two clocks around dispatch, synchronized elapsed has one post-loop
  completion call, and warmup/reset/serialization remain outside those
  intervals. Tests retain replay-count-major positive raw samples and mean
  ns/operation metrics; cover invalid clocks, identity/dimension drift, reset,
  dispatch, synchronization, provenance, serialization, and atomic-publication
  failures. CUDA absence and preflight reset incapability publish complete
  unavailable artifacts without timer, operation, synchronization, or reset
  calls. `resident_benchmark_cuda_support_test.py` additionally covers mutable
  primary/RNG snapshot restoration, empty registries, validation order, and
  propagated reset errors. The opt-in CUDA row asserts artifact structure, never
  a speedup threshold.
- **P3:** Parser tests use checked-in bounded text/JSON fixtures for supported
  profiler exports. Cover unit conversion, duplicate invocations, unattributed
  kernels, missing counters, unsupported versions, bounded errors, and stable
  aggregation. Process-runner tests mock `nsys` and `ncu` and cover argument
  vectors, `shell=False`, version probes, timeouts, exit status, bounded output,
  and safe export paths. A separate opt-in smoke test invokes both installed
  binaries around one bounded CUDA workload, exports real reports, and passes
  those reports through the production parser. It may pass or report explicit
  unavailable evidence; a schema or parser mismatch for the selected installed
  versions is a failure.
- **P4:** Unit tests cover contribution reconciliation, deterministic ranking,
  ties, low-confidence/missing evidence, machine-bound wording, and rejection
  of recommendations that alter scientific or ownership contracts.
- **P5:** Documentation contract tests verify commands, machine and workload
  bounds, raw evidence links, limitations, and the T7/E8-F7 reconciliation;
  strict MkDocs validates rendering and links.

Likely test locations are `particula/gpu/tests/profiling_support_test.py`,
`particula/gpu/tests/benchmark_helpers_test.py`, and opt-in rows in
`particula/gpu/tests/benchmark_test.py`.

## Focused Assertion Checks

Focused fix checks run without coverage:

```bash
pytest particula/gpu/tests/profiling_support_test.py \
  particula/gpu/tests/benchmark_helpers_test.py -q
pytest particula/gpu/tests/benchmark_test.py --benchmark \
  -k "resident and (launch or profile)" -v -s
pytest particula/gpu/tests/profiling_smoke_test.py --benchmark \
  -m "warp and cuda" -q -Werror
```

The second and third commands are CUDA-only and may pass or cleanly report an
unavailable prerequisite. A skip is not a measurement and must remain an
unavailable artifact row. Neither command may route to Warp CPU. The smoke test
runs vendor profilers only to verify executable, export, and parser integration;
its overhead-tainted timings are not benchmark thresholds.

## Coverage and Final Validation

A focused target with `--cov` is invalid comprehensive evidence. Focused checks
are coverage-disabled assertion checks; inability to obtain full-package
coverage from them is a validation-infrastructure issue, not a feature failure.
After focused checks pass, run the untargeted repository suite, which supplies
repository-configured full-package coverage and its normal threshold:

```bash
.opencode/tools/run_pytest.py
mkdocs build --strict
```

No coverage threshold is lowered. Required profiler rows that cannot run are
recorded unavailable and keep the profiling closeout unshipped rather than being
inferred as passing.
