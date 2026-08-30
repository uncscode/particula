# Testing Strategy

Every phase ships co-located `*_test.py` coverage. Default tests validate
records, parsing, timing control flow, analysis, and documentation without CUDA
or NVIDIA tools. Real timings and profiles remain opt-in evidence, never stable
assertion thresholds.

## Per-Phase Approach

- **P1:** Unit tests cover exact types, schema versions, canonical ordering,
  workload IDs, machine metadata allow-lists, safe paths, checksums, invalid
  dimensions/units, and explicit unavailable records.
- **P2:** Timer and synchronizer spies prove setup/capture/warmup are excluded,
  captured and uncaptured paths use identical fixture identities and step
  counts, each raw sample is retained, and absent CUDA skips without fallback.
  A real CUDA row records samples but asserts structure rather than speedup.
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
