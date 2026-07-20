# Testing Strategy

- **P1 (shipped, issue #1372):**
  `particula/tests/gpu_coagulation_docs_test.py` uses only the standard library
  and asserts the canonical import, executable/deferred mask boundary, exact
  support/non-support claims, ownership and preflight language, required Warp
  CPU commands, optional CUDA wording, and resolving guide links. The legacy
  exact command-matrix assertions in
  `particula/tests/condensation_latent_heat_docs_test.py` cover the new rows.
- **P2 (shipped, issue #1373):**
  `particula/gpu/tests/gpu_coagulation_direct_example_test.py` covers CPU-safe
  import, forced/unavailable-Warp no-execution behavior, exact lazy-import
  order, deterministic disabled output, shared sidecar identity, two-call RNG
  initialization/reuse, and failure propagation. Guarded Warp CPU cases verify
  RNG advancement, valid final-call pairs, mass/charge conservation, unchanged
  inactive slots, and zero/one-active early returns; CUDA remains unrequired.
- **P3 (shipped, issue #1374):**
  `particula/tests/gpu_coagulation_docs_test.py` uses only the standard library
  to require one matching E5 inventory in each roadmap, all nine unique child
  IDs, exact canonical rows and artifact targets, resolving local links, and no
  stale placeholder. It rejects duplicate
  records and duplicated artifact links without requiring GPU hardware.
- **P4 (shipped, issue #1375):** Completed the fail-closed closeout gate with
  E5-F7's focused parity/stochastic/conservation matrix, both new test modules,
  the example, docs validation, fast tests, and lint. Gate fixtures assert that
  failed prerequisites block the successful status transition and that the
  final state depends on authoritative feature and phase status, issue and
  commit references, resolving artifact paths, focused command results, and
  required Warp CPU evidence; issue closure alone is never sufficient.

Tests ship in the same phase as changes and use `*_test.py`. Coverage thresholds
must not be lowered; changed executable code must maintain at least 80% coverage.
Warp CPU is required whenever Warp is installed. CUDA is additive local/manual
evidence and must skip cleanly when unavailable. Stochastic checks use bounded
aggregate behavior, never exact CPU/GPU pair replay.

## Test Boundaries and Assertions

- Keep documentation and example regression tests in their named `tests/`
  directories; use `particula/integration_tests/` only if the direct example
  needs an end-to-end package workflow that cannot be isolated in its module.
- Use explicit `np.float64` fixtures and `numpy.testing.assert_allclose` with
  stated `rtol` and `atol` for deterministic Warp CPU observations. Assert
  mass and charge conservation independently so an aggregate inventory result
  cannot hide either regression.
- For P2, parameterize the no-Warp and Warp CPU branches. The CUDA parameter is
  optional and must use the repository's clean unavailable-device skip, not an
  expected failure. Preserve caller-buffer identity and persistent RNG state
  checks in the same test that exercises the example.
- P4 was a release-gate validation phase, not a substitute for feature tests.
  It ran the P1--P3 test modules and E5-F7 evidence without moving their
  assertions into a later change; gate fixtures cover failed and complete
  prerequisites.

## Focused Verification Commands

```bash
pytest particula/tests/gpu_coagulation_docs_test.py -q -Werror
pytest particula/gpu/tests/gpu_coagulation_direct_example_test.py -q -Werror
pytest particula/gpu/kernels/tests/coagulation_validation_test.py -q -Werror
pytest particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py -q -Werror
pytest particula/gpu/kernels/tests/coagulation_test.py -q -Werror
```

Run the optional CUDA-marked cases only on a CUDA-capable host. Follow the
focused runs with the repository regression and lint gates before allowing the
P4 closeout transition.
