# Testing Strategy

Every phase ships co-located `*_test.py` coverage. Benchmark timing is
supplemental evidence, not an assertion threshold, and remains opt-in.

## Per-Phase Approach

- **P1 (delivered, issue #1581):** Host-only default-collection tests in
  `particula/execution/tests/resident_benchmark_support_test.py` validate frozen
  records, metadata, stable schema-versioned JSON ordering/round trips, raw
  sample summaries, malformed-input rejection, path/symlink safety, and atomic
  writer failure handling. They verify that importing the support module does
  not load Warp and do not probe CUDA or allocate device memory.
- **P2 (delivered, issue #1582):** Default-collection tests in
  `resident_benchmark_support_test.py` and
  `resident_benchmark_cuda_support_test.py` cover bounded count rejection before
  callbacks, alternating warmup/sample order, no warmup synchronization, one
  synchronization per measured operation, schema-v1 decode/schema-v2 round
  trip, isolated artifact persistence, lazy Warp import, binding drift
  rejection, and setup/capture timing boundaries. The opt-in
  `test_resident_captured_replay_comparison` row in `benchmark_test.py` is
  CUDA-only; it records two equal nonempty sample sets and validates summaries
  when CUDA/native capture qualifies, otherwise cleanly skips.
- **P3 (delivered, issue #1583):** Default-collection host tests in
  `resident_benchmark_support_test.py` cover the exact 1/10/100/1000 matrix,
  explicit axes, exact requested/actual dimensions, equality and over-budget
  boundaries, malformed estimates/availability, import isolation, and no probe
  for invalid or over-budget cases. CUDA-seam and opt-in consumer tests cover
  exact nondefault dimension forwarding, one availability probe across eligible
  rows, structured unavailable outcomes, one binding per approved row, cleanup,
  one aggregate writer call, and no CPU/Warp-CPU fallback.
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
pytest particula/execution/tests/resident_benchmark_support_test.py -q --no-cov
pytest particula/execution/tests/resident_benchmark_cuda_support_test.py -q --no-cov
pytest particula/execution/tests/multi_box_loop_test.py -q --no-cov
pytest particula/execution/tests/captured_full_loop_test.py -q --no-cov
pytest particula/execution/tests/captured_full_loop_test.py -q --no-cov
pytest particula/gpu/tests/benchmark_safety_test.py --benchmark -q --no-cov
pytest particula/gpu/tests/benchmark_test.py --benchmark \
  -k resident -v -s --no-cov
```

The CUDA benchmark command may pass or cleanly skip only before the resident
artifact path is entered. Once entered, unavailable CUDA/native capture is a
structured row. It must never fall back to Warp CPU, and nonexecuted rows are
not inferred as measurements.

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
