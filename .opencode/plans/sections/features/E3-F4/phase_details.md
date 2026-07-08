# Phase Details

## E3-F4-P1: Decide low-level GPU kernel import path and public surface with regression tests

Size: S

Depends on: E3-F1 finalizing caller-visible persisted `rng_states` guidance and
maintainer direction that the direct quick-start path should come from
`particula.gpu.kernels` unless narrowly expanded with explicit export tests.

Decide whether the stable direct-kernel user path remains
`particula.gpu.kernels` or whether the two step functions should also be
re-exported from `particula.gpu`. Keep the decision narrow: direct low-level
kernels only, no backend selector and no broad exposure of internal Warp launch
functions. Add an initial regression test capturing the selected path.

Test coverage in this phase:

- Add or extend `particula/gpu/tests/kernel_exports_test.py` to assert the
  selected import path resolves without launching Warp kernels.
- If top-level exports are chosen, assert `particula.gpu.__all__` exposes only
  `condensation_step_gpu` and `coagulation_step_gpu` from this decision.
- If `.kernels` remains the supported path, add a negative regression assertion
  that undocumented top-level imports are absent so the public surface does not
  drift silently.

Deliverables:

- Documented import-path decision in the implementation PR and plan updates.
- Test coverage proving the selected import path resolves.
- If top-level exports are chosen, update `particula/gpu/__init__.py` and
  `__all__` for only `condensation_step_gpu` and `coagulation_step_gpu`.

## E3-F4-P2: Add stable import/export tests for direct condensation and coagulation kernels

Size: S

Depends on: E3-F4-P1 choosing the supported import surface so regression tests
lock the exact public path and rejection boundaries before docs examples depend
on them.

Add dedicated tests, likely `particula/gpu/tests/kernel_exports_test.py`, or
extend the existing top-level export tests. These tests should assert selected
imports work when Warp is unavailable and avoid launching kernels unless needed.

Test coverage in this phase:

- Add parametrized import tests for both `condensation_step_gpu` and
  `coagulation_step_gpu` in `particula/gpu/tests/kernel_exports_test.py`.
- Cover the supported module path plus rejected/raw internal symbol imports when
  the API decision excludes them.
- Keep the test module warning-clean under `pytest -Werror` and runnable on
  CPU-only environments.

Deliverables:

- Regression tests for direct condensation and coagulation imports.
- `__all__` assertions for any chosen public module.
- Negative/guard assertions preventing accidental dependence on raw internal
  kernel symbols where the API decision excludes them.

## E3-F4-P3: Create runnable direct-kernel quick-start example with explicit transfer helpers

Size: S

Depends on: E3-F4-P1 and E3-F4-P2 settling the supported import path and export
guardrails, plus E3-F1 confirming the repeated-call RNG usage that the
coagulation snippet is allowed to demonstrate.

Create a runnable docs example following the existing examples style. The
example should build minimal `ParticleData` and `GasData`, gate GPU execution
with `WARP_AVAILABLE`, transfer explicitly, call both low-level kernels, and
transfer results back explicitly at the end. Use `gpu_context` where it clarifies
particle transfer, while keeping gas/environment transfer explicit.

Test coverage in this phase:

- Add or extend `particula/gpu/tests/data_containers_example_test.py` with a
  smoke test that imports the example module, verifies the no-Warp path exits
  cleanly, and exercises `device="cpu"` when Warp is available.
- Assert the example uses explicit `to_warp_*` and `from_warp_*` helper calls so
  the documented transfer boundary stays visible to reviewers.
- If the example introduces helper functions, keep them covered in the same test
  module rather than a later follow-up phase.

Deliverables:

- New or updated quick-start example under `docs/Examples/`.
- Demonstrated condensation direct call.
- Demonstrated coagulation direct call using explicit buffers/RNG state once
  compatible with `E3-F1` persistence.
- No hidden transfer or backend-selection helpers.

## E3-F4-P4: Document troubleshooting and validate docs quick-start behavior

Size: S

Depends on: E3-F4-P3 producing the stable quick-start artifact. Follow any
shared wording updates from E3-F5 after marker/helper policy lands, but do not
let that later wording pass block the example itself.

Add user-facing troubleshooting and smoke tests for the quick-start. The final
phase verifies a new user can run the example on CPU-backed Warp and skip
cleanly when Warp/CUDA is unavailable.

Test coverage in this phase:

- Re-run `particula/gpu/tests/kernel_exports_test.py` and
  `particula/gpu/tests/data_containers_example_test.py` after documentation
  updates so the documented import path and runnable example stay aligned.
- Run focused kernel regression checks in
  `particula/gpu/kernels/tests/condensation_test.py` and
  `particula/gpu/kernels/tests/coagulation_test.py` when troubleshooting text
  changes example inputs or validation guidance.
- Treat CUDA as optional: verify CUDA branches skip with clear reasons when
  unavailable instead of failing default CI.

Deliverables:

- Troubleshooting for missing Warp, missing CUDA, device mismatch, mixed
  `environment=` plus scalar inputs, and transfer-boundary mistakes.
- Example smoke tests using existing docs-example test patterns.
- Documentation updates linking the quick-start from relevant GPU feature docs,
  including developer-facing GPU roadmap/foundation guidance updates.
