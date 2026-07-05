# Testing Strategy

## Co-located Testing Policy

Each implementation phase ships tests next to the code it changes. There is no
standalone testing phase. Fast GPU tests should use the existing Warp CPU path
and opt into CUDA only through the repository's availability helpers.

## Required Test Categories

- **Scalar compatibility:** Existing scalar calls to `condensation_step_gpu` and
  `coagulation_step_gpu` continue to pass without requiring environment objects.
- **Keyword-only extension:** `environment=` remains keyword-only so positional
  scalar callers do not gain a new positional dependency.
- **P1 mixed-input rejection:** Passing explicit `environment` together with any
  scalar `temperature` or `pressure` raises a stable early `ValueError`.
- **P1 explicit-environment rejection:** Passing `temperature=None`,
  `pressure=None`, and `environment=...` raises the phase-scoped P1
  `ValueError` until later phases implement execution.
- **Pre-launch short-circuiting:** Contract failures occur before
  `get_dynamic_viscosity(...)`, `get_molecule_mean_free_path(...)`,
  `_ensure_volume_array(...)`, or any Warp launch.

## Test Locations

- Condensation behavior and validation:
  `particula/gpu/kernels/tests/condensation_test.py`.
- Coagulation behavior and validation:
  `particula/gpu/kernels/tests/coagulation_test.py`.
- Container/conversion behavior stays covered by existing helpers; P1 reuses
  `EnvironmentData` plus `to_warp_environment_data(...)` inside the co-located
  kernel tests instead of adding new conversion test files.

## Verification Commands

- `pytest particula/gpu/kernels/tests/condensation_test.py -q`
- `pytest particula/gpu/kernels/tests/coagulation_test.py -q`
- `ruff check particula/ --fix && ruff format particula/ && ruff check particula/`
