# Testing Strategy

Every phase ships co-located `*_test.py` coverage. Benchmark timing is
supplemental evidence, not an assertion threshold, and remains opt-in.

## Per-Phase Approach

- **P1:** Fast unit tests validate case schemas, metadata, stable JSON ordering,
  raw-sample summaries, path safety, and malformed/overflowing input rejection.
- **P2:** Unit tests spy on prepared/captured routing, warmup exclusion, explicit
  synchronization, exact fixture identity, and separation of setup from replay.
  One opt-in CUDA row records real uncaptured/captured samples.
- **P3:** Parametrized unit tests verify the required box-count matrix and all
  secondary axes. Budget boundaries and unavailable CUDA/capture rows must be
  deterministic and must never dispatch a CPU fallback.
- **P4:** Formula tests cover each memory category, zero dimensions, inactive
  fixed capacity, E8-F3 total reconciliation, communication alternatives,
  checkpoint scenarios, projected tape scaling, and checked integer arithmetic.
- **P5:** Integration tests exercise observed-memory probe availability and a
  representative CUDA fixture. Assertions cover schema and nonnegative deltas,
  not a machine-independent allocator ratio.
- **P6:** Documentation contract tests verify exact commands, provenance,
  limitations, and unavailable rows; strict MkDocs validates links/rendering.

Likely locations are `particula/execution/tests/` for resident support and
integration tests and `particula/gpu/tests/benchmark_helpers_test.py` plus
`benchmark_test.py` for fast helper and opt-in CUDA coverage.

## Commands and Coverage

Focused fix checks are assertion-only and coverage disabled:

```bash
pytest particula/gpu/tests/benchmark_helpers_test.py \
  particula/gpu/tests/benchmark_safety_test.py -q
pytest particula/execution/tests/ -q -k "benchmark or memory_budget"
pytest particula/gpu/tests/benchmark_test.py --benchmark \
  -k "resident and (scaling or memory)" -v -s
```

The CUDA benchmark command may pass or cleanly skip. It must never fall back to
Warp CPU, and skipped/unavailable rows are not inferred as measurements.

A focused target with `--cov` is invalid comprehensive evidence. Focused checks
must run without coverage; inability to meet full-package coverage from those
targets is a validation-infrastructure mistake, not a feature failure. Final
validation uses the untargeted repository runner, which supplies configured
full-package coverage and its normal threshold:

```bash
.opencode/tools/run_pytest.py
mkdocs build --strict
```

No threshold is lowered. If a required command cannot run, record it as
unavailable and keep the applicable evidence row unshipped.
