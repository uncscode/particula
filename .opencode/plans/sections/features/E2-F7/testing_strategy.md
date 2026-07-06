# E2-F7 Testing Strategy

## Co-located Testing Policy

Each phase that adds helper functions, prototypes, or production hooks must add
tests in the same phase. There is no standalone testing phase.

## Fast Tests

- The executable focused coverage currently lives in
  `particula/gpu/kernels/tests/_condensation_test_support.py`; the wrapper
  files `condensation_test.py` and `condensation_stiffness_test.py` are stub
  placeholders in the current tree and must not be treated as the runnable
  verification target unless they are first restored with real tests.
- Reuse focused selectors in `_condensation_test_support.py` for:
  - stress-case fixture construction,
  - fixed-shape and dtype metadata validation,
  - scalar `temperature`/`pressure` coverage,
  - accepted direct `(n_boxes,)` Warp-array environment inputs,
  - threshold-boundary classification semantics,
  - particle-only caveat handling,
  - zero-mass stability,
  - finite/non-negative post-step checks, and
  - pre-launch validation failures that short-circuit before kernel execution.
- Keep tests compatible with Warp CPU execution by default.
- Use CUDA-specific execution only behind existing availability guards.

## Target Test and Evidence Files

- `particula/gpu/kernels/tests/_condensation_test_support.py` for all fast
  P1-P4 executable coverage, including the shipped P3 test-local candidate
  evidence and the P4 rerun selectors.
- `particula/integration_tests/condensation_particle_resolved_test.py` remains
  future coverage if a gas-coupled production hook lands in a later phase.
- `particula/gpu/kernels/tests/condensation_test.py` and
  `particula/gpu/kernels/tests/condensation_stiffness_test.py` are currently
  non-executable wrappers only.
- `docs/Features/Roadmap/condensation-stiffness-study.md` now records the
  baseline case catalog, metric vocabulary, the shipped P2 measured-results
  table synchronized with the recorded timestep grid, and the shipped P3
  candidate evidence plus graph-capture/autodiff notes.
- `docs/Features/Roadmap/warp-autodiff-limitations.md` only for cross-links and
  documented constraints in P4; do not move executable assertions into docs.

## Reference Tests

- Reuse CPU reference functions from `mass_transfer.py` and
  `mass_transfer_utils.py` to validate rate and limiter expectations.
- Reuse patterns from staggered stability and mass-conservation tests, but avoid
  requiring slow benchmarks for default CI.

## Metrics to Assert

- Particle masses remain finite and non-negative.
- Explicit fractional mass changes stay below documented thresholds for a
  timestep classified as stable.
- Zero-initial-mass bins remain unchanged when classified stable.
- Buffer shapes remain fixed and preallocated buffers are reused.
- Documentation and helper outputs accurately mark GPU particle-only behavior
  versus future gas-particle conservation work.

Map those assertions to phases: P1 now covers fixture/metric correctness,
environment-input contract coverage, and negative validation paths. P2 now adds
recorded-grid stability evidence, exact timestep-order checks, caller-owned
buffer reuse/overwrite assertions, unchanged-gas assertions, and scalar-vs-Warp
environment-input mode coverage. P3 now adds deterministic
`fixed_count_substeps_4` and `asymptotic_relaxation` comparisons, reusable
fixed-shape scratch/buffer assertions, finite/non-negative candidate checks,
CPU-reference tolerance checks, and explicit-baseline error-bound assertions,
while leaving gas-coupled production/conservation coverage deferred. P4 remains
the documentation-only exception that must rerun the same focused fast tests
before publishing conclusions.

## Slow/Benchmark Coverage

- Extended timestep sweeps may be marked slow or kept as documentation
  generation scripts if they are too expensive for default CI.
- Slow tests should report enough context to reproduce the stiffness map but
  should not gate normal development unless runtime remains small.

If wrapper files are restored later, keep at least one representative fast
assertion in a real `*_test.py` entry point that proves the recommendation is
tied to executable behavior rather than report-only analysis.

## Verification Commands

```bash
pytest particula/gpu/kernels/tests/_condensation_test_support.py -k "stiffness or candidate" -v
pytest particula/gpu/kernels/tests/_condensation_test_support.py -k "condensation_step_gpu or apply_mass_transfer_kernel or condensation_skips_inactive_particles or condensation_multi_species_parity" -v
pytest particula/gpu/kernels/tests/_condensation_test_support.py -k "rejects or short_circuit or mismatch or invalid_scalar_domains or raises_value_error" -v
pytest particula/dynamics/condensation/tests/staggered_stability_test.py -m slow -v
pytest particula/integration_tests/condensation_particle_resolved_test.py -v
ruff check particula/ --fix && ruff format particula/ && ruff check particula/
```
