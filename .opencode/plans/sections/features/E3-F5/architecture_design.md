# Architecture and Design

## Design Principles

- Centralize policy in pytest configuration and small GPU test helpers rather
  than scattering device decisions across individual test modules.
- Keep CUDA optional. The policy should never require a CUDA device for default
  CI or local CPU-only development.
- Prefer explicit markers over implicit naming conventions so reviewers can
  select or deselect device classes with `pytest -m`.
- Preserve explicit CPU/GPU transfer boundaries; helpers must not hide device
  synchronization, copy data implicitly, or silently fall back from CUDA to CPU.

## Proposed Marker Vocabulary

- `warp`: tests that require the `warp` package and should skip when it is not
  installed.
- `cuda`: tests that require a CUDA device specifically and must skip cleanly
  when CUDA is unavailable.
- `gpu_parity`: tests comparing CPU, Warp CPU, and/or CUDA behavior.
- `stochastic`: tests where pass/fail is based on aggregate statistical behavior
  or tolerance bands rather than exact deterministic equality.

E3-F5-P1 shipped these exact names. The implementation centralizes the strings
in `particula/conftest.py::PYTEST_MARKER_LINES` and mirrors them verbatim in
`pyproject.toml`, with regression tests asserting parity between the two.

## Option Surface

- `--benchmark` remains the only registered pytest option.
- No CUDA-selection or device-policy CLI option was added in P1.
- Default collection behavior remains registration-only for `warp`, `cuda`,
  `gpu_parity`, and `stochastic`; benchmark gating is still the only hook that
  modifies collected items.

## Helper Structure

The shipped helper structure stayed intentionally small. `E3-F5-P4` reused the
existing `particula/gpu/tests/cuda_availability.py` surface rather than adding a
new policy module:

- `warp_devices(wp) -> list[str]`: existing behavior, still returns `['cpu']`
  plus CUDA when available.
- `CUDA_SKIP_REASON`: standardized CUDA-only skip message reused by migrated
  suites and helper coverage.
- No `device_policy.py`, parameter-wrapper helper, or new tolerance helper was
  introduced; the implementation kept local fixtures where that preserved test
  readability and wrapper export behavior while still standardizing on
  `warp_devices(wp)`.

## Tolerance Model

- Deterministic floating parity: use explicit `rtol`/`atol` with
  `numpy.testing.assert_allclose` or `pytest.approx`.
- Conservation checks: use tight tolerances appropriate for mass/number
  accounting, preserving existing examples around `rtol=1.0e-12` where valid.
- Stochastic coagulation: aggregate over seeds or steps, compare expected means
  with `3 * sqrt(mean)`-style bands or established tolerance arrays, and avoid
  exact per-seed equality across devices.

## Cross-feature Fit

E3-F5 should consume E3-F1 seed-once RNG expectations and E3-F2 mixed-scale
sampling evidence. It should not re-implement those algorithms; it should make
their validation pattern reusable and reviewable.

## Shipped P4 Test-surface Design

- Directly discoverable Warp suites now declare `pytestmark = pytest.mark.warp`
  at module scope in `coagulation_test.py`, `environment_test.py`,
  `conversion_test.py`, `condensation_test.py`, and
  `condensation_stiffness_test.py`, with `_condensation_test_support.py` also
  marked for support-module readability.
- Wrapper/export structure was preserved for condensation coverage by keeping
  wrapper-level Warp marks and `support.device` re-exports, so support-backed
  tests still collect through the discoverable wrapper modules.
- Deterministic cross-device comparisons are now marked with `gpu_parity` in
  representative coagulation, condensation-support, and conversion tests.
- Aggregate/statistical coagulation checks are now marked with `stochastic`
  instead of being inferred indirectly from file names or comments.
- CUDA-only readback or wrong-device checks are marked narrowly with `cuda`
  so Warp CPU remains the default runnable path for mixed suites.
