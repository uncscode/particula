# Testing Strategy

- **P1:** `particula/tests/gpu_coagulation_docs_test.py` asserts the canonical
  import, exact support/non-support claims, ownership language, required Warp
  CPU commands, optional CUDA wording, and resolving guide links.
- **P2:** `particula/gpu/tests/gpu_coagulation_direct_example_test.py` imports
  without Warp, exercises the forced no-Warp branch, checks lazy imports and
  deterministic output, and on Warp CPU verifies execution, buffer identity,
  persistent-RNG advancement, valid pairs, and mass/charge conservation.
- **P3:** docs tests require exactly one E5 row, all nine unique child IDs,
  matching artifact targets, no stale E5 placeholder, and valid local links.
- **P4:** run E5-F7's focused parity/stochastic/conservation matrix, both new
  test modules, the example, docs validation, fast tests, and lint. Assert that
  failed prerequisites preserve E5 active/Epic F pending before testing the
  successful status transition. Gate closeout on authoritative feature and
  phase status, issue and commit references, resolving artifact paths, focused
  command results, and required Warp CPU evidence; issue closure alone is never
  sufficient.

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
- P4 is a release-gate validation phase, not a substitute for feature tests:
  it runs the P1--P3 test modules and E5-F7 evidence without moving their
  assertions into a later change. Its status-transition test must cover both a
  failed-prerequisite case and the all-prerequisites-passed case.

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
