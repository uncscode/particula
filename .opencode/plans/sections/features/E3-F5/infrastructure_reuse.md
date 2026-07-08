# Infrastructure Reuse

## Existing Test Hooks

- Reuse `particula/conftest.py` for pytest option registration, marker
  registration, and collection-time skip behavior. It already handles the
  `--benchmark` option and `benchmark` marker.
- Mirror new marker definitions in `pyproject.toml` under
  `[tool.pytest.ini_options].markers`, following the existing `slow`,
  `performance`, and `benchmark` entries.
- Extend or mirror the style of `particula/tests/benchmark_option_test.py` for
  hook-level regression coverage.

## Existing Warp Device Helpers

- Reuse `particula/gpu/tests/cuda_availability.py`, which provides:
  - `cuda_available(wp)` with warning suppression.
  - `warp_devices(wp)` returning `['cpu']` plus `['cuda']` when available.
- Keep `pytest.importorskip('warp')` as the module-level pattern for tests that
  require Warp.
- Preserve local CUDA-specific skips where a test truly requires CUDA-only
  behavior, but route them through the standardized helper where possible.

## Existing GPU Test Targets

- `particula/gpu/kernels/tests/coagulation_test.py` contains stochastic parity,
  RNG state, and conservation examples that should receive the policy markers
  and helper usage.
- `particula/gpu/kernels/tests/_condensation_test_support.py` and
  `condensation_test.py` provide the condensation device-fixture pattern.
- `particula/gpu/kernels/tests/environment_test.py` contains CUDA optional skip
  examples and wrong-device validation.
- `particula/gpu/tests/conversion_test.py` and `data_containers_example_test.py`
  provide CPU/GPU transfer-boundary parity patterns.

## Documentation Targets

- `.opencode/guides/testing_guide.md` is the primary internal policy document.
- `docs/Features/Roadmap/data-oriented-gpu.md` should record the E3-F5
  policy and tolerance outcome.
- `docs/Features/data-containers-and-gpu-foundations.md` should only be touched
  if user-facing device validation guidance needs clarification.
