# Implementation Tasks

## Phase E3-F4-P1

- Review `particula/gpu/__init__.py` and `particula/gpu/kernels/__init__.py` to
  map the current export surface before editing any docs examples.
- Choose the public direct-kernel import path and record the decision in
  `docs/Features/Roadmap/data-oriented-gpu.md` before touching broader examples.
- If top-level re-export is selected:
  - Import `condensation_step_gpu` and `coagulation_step_gpu` in
    `particula/gpu/__init__.py`.
  - Add both names to `__all__`.
- If `.kernels` remains selected:
  - Leave top-level exports focused on transfer/context helpers.
  - Document `.kernels` as the low-level path.
- Add or update a regression test in the new
  `particula/gpu/tests/kernel_exports_test.py` (or one existing export-focused
  module if it already covers this surface) for the selected path only; keep the
  code change to small `__init__` edits plus one focused import assertion block
  around `condensation_step_gpu` and `coagulation_step_gpu`.

## Phase E3-F4-P2

- Add `particula/gpu/tests/kernel_exports_test.py` or extend one existing
  export-focused test module as the single home for direct-kernel import checks.
- Assert `condensation_step_gpu` and `coagulation_step_gpu` import from the
  selected public module.
- Assert `__all__` coverage for selected public exports and a negative case for
  any raw internal symbol intentionally excluded from the public API.
- Ensure tests do not require CUDA, do not launch kernels, and still pass when
  Warp is unavailable by checking import surface only; keep this phase to one
  small test module rather than scattering import assertions across unrelated
  GPU tests.

## Phase E3-F4-P3

- Create one canonical direct-kernel quick-start under `docs/Examples/`, such
  as `docs/Examples/gpu_direct_kernels_quick_start.py`, and reuse that same path
  in smoke tests and cross-links.
- Build small deterministic `ParticleData` and `GasData` fixtures in the example
  file itself so users can copy the minimal direct-kernel setup without chasing
  hidden helpers.
- Gate execution with `WARP_AVAILABLE` and provide a useful no-Warp message.
- Use explicit transfer helpers and `gpu_context` where appropriate.
- Call `condensation_step_gpu` with compatible temperature/pressure or
  environment inputs.
- Call `coagulation_step_gpu` with explicit caller-owned state/buffers where
  needed.
- Transfer results back explicitly at the end for a printed or asserted summary,
  ideally via one `run_example()` helper and a pair of small `_build_*` fixture
  helpers so the example remains reviewable; keep the example to one
  condensation step plus one coagulation step so the documentation phase stays
  S-sized.

## Phase E3-F4-P4

- Add smoke tests modeled after `data_containers_example_test.py`, ideally in a
  dedicated docs-example test file such as
  `particula/gpu/tests/gpu_direct_kernels_example_test.py` adjacent to the
  existing GPU example tests.
- Test the no-Warp path and Warp CPU path with import-and-run coverage only;
  keep CUDA guidance optional in docs rather than required in the smoke test.
- Add optional CUDA guidance without making CUDA mandatory in CI.
- Update feature docs/roadmap links to point to the quick-start.
- Add troubleshooting entries for installation, CUDA, device mismatch, mixed
  `environment=` inputs, and explicit transfer-helper mistakes.
