# Testing Strategy

## Fast Validation

- Run focused fast tests for any changed benchmark helpers, for example:
  `pytest particula/gpu/tests/benchmark_helpers_test.py -q`.
- Run markdown or documentation validation required by the documentation files
  touched during implementation.
- If paired notebook content under `docs/Theory/nvidia-warp/examples/` changes,
  validate the paired `.py` source and sync/validate the notebook using the
  repository notebook tooling.

## Benchmark Validation

- Run `pytest particula/gpu/tests/benchmark_test.py --benchmark -v -s` on
  CUDA-capable hardware when available.
- Record clean skip behavior when CUDA or benchmark opt-in support is absent.
- Preserve benchmark markers (`slow`, `performance`, `benchmark`) so normal CI
  and local fast test runs remain unaffected.

## Co-located Testing Policy

This feature is primarily documentation and benchmark evidence. If any helper
functions are added or changed, their unit tests must ship in the same phase
under the existing GPU test structure. Do not create a standalone testing phase.

## Non-goals for Tests

- Do not require CUDA in default test runs.
- Do not turn performance thresholds into brittle CI gates unless a narrow,
  deterministic helper behavior is being tested.
- Do not add graph-capture or production optimization tests in this feature.
