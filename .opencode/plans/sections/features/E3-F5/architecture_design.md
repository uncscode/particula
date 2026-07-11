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

Extend `particula/gpu/tests/cuda_availability.py` or introduce
`particula/gpu/tests/device_policy.py` with lightweight helpers such as:

- `warp_devices(wp) -> list[str]`: existing behavior, still returns `['cpu']`
  plus CUDA when available.
- `skip_if_cuda_unavailable(wp, reason: str) -> None`: standardized CUDA-only
  skip message.
- `warp_device_params(wp)`: optional pytest parameter wrapper if markers/ids are
  easier to apply at parametrization time.
- Named tolerance constants or helper functions for stochastic coagulation bands
  only if they reduce duplication without obscuring assertions.

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
