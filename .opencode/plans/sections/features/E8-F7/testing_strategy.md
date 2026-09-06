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
- **P3 (delivered):** `profiling_support_test.py` uses bounded fixtures and a
  mock runner to cover strict version/CSV/JSON parsing, unit conversion,
  attribution, aggregation, command vectors, `shell=False`, timeouts, bounded
  diagnostics, failures, and contained paths. `profiling_workload_runner_test.py`
  covers argv rejection before CUDA imports, exact qualified-worker call order,
  unavailable handling, and one-time cleanup. `profiling_smoke_test.py` adds a
  hardware-free composition seam plus an opt-in `benchmark`/`warp`/`cuda` smoke
  row that profiles the closed worker once per independently qualified tool.
  Missing prerequisites skip without fallback; post-qualification collection,
  export, schema, parser, or mapping failures fail.
- **P4:** Unit tests cover contribution reconciliation, deterministic ranking,
  ties, low-confidence/missing evidence, machine-bound wording, and rejection
  of recommendations that alter scientific or ownership contracts.
- **P5:** Documentation contract tests verify commands, machine and workload
  bounds, raw evidence links, limitations, and the T7/E8-F7 reconciliation;
  strict MkDocs validates rendering and links.

P3 test locations are `particula/gpu/tests/profiling_support_test.py`,
`particula/gpu/tests/profiling_workload_runner_test.py`, and
`particula/gpu/tests/profiling_smoke_test.py`.

## Focused Assertion Checks

Focused fix checks run without coverage:

```bash
pytest particula/gpu/tests/profiling_support_test.py \
  particula/gpu/tests/profiling_workload_runner_test.py -q --no-cov
pytest particula/gpu/tests/profiling_smoke_test.py --benchmark \
  -m "warp and cuda" -q --no-cov
```

The second command is CUDA-only and may pass or cleanly skip an unavailable
prerequisite. A skip is not a measurement. Neither command may route to Warp
CPU. The smoke test verifies executable, export, and parser integration only;
it is not a benchmark threshold or published profile result.

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
