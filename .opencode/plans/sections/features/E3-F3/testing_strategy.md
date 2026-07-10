# Testing Strategy

## Fast Validation

- Run focused fast tests for any changed benchmark helpers, for example:
  `pytest particula/gpu/tests/benchmark_helpers_test.py -q`.
- Keep fast helper coverage structural and bounded: assert deterministic mixed-
  scale coagulation fixture shapes/spread, positive active concentrations,
  routing isolation between `_make_particle_data(...)` and
  `_make_coagulation_particle_data(...)`, benchmark result recording, and
  persistent RNG-state reuse semantics.
- Run markdown or documentation validation required by the documentation files
  touched during implementation.
- If paired notebook content under `docs/Theory/nvidia-warp/examples/` changes,
  validate the paired `.py` source and sync/validate the notebook using the
  repository notebook tooling.

## Benchmark Validation

- Run `pytest particula/gpu/tests/benchmark_test.py --benchmark -v -s` on
  CUDA-capable hardware when available.
- For this shipped phase, capture the benchmark artifact path and the compact
  single-box vs multi-box coagulation timing summary used in the roadmap note.
- Preserve benchmark markers (`slow`, `performance`, `benchmark`) so normal CI
  and local fast test runs remain unaffected.

## Co-located Testing Policy

This feature is primarily benchmark evidence and roadmap documentation. Helper
changes must ship with same-phase tests under the existing GPU test structure;
for P1 this coverage lives in `particula/gpu/tests/benchmark_helpers_test.py`.

## Non-goals for Tests

- Do not require CUDA in default test runs.
- Do not turn performance thresholds into brittle CI gates unless a narrow,
  deterministic helper behavior is being tested.
- Do not add graph-capture or production optimization tests in this feature.
