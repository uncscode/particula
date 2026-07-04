# Testing Strategy

## Co-located Testing Policy

Each implementation phase ships tests next to the code it changes. There is no
standalone testing phase. Fast GPU tests should use the existing Warp CPU path
and opt into CUDA only through the repository's availability helpers.

## Required Test Categories

- **Scalar compatibility:** Existing scalar calls to `condensation_step_gpu` and
  `coagulation_step_gpu` continue to pass without requiring environment objects.
- **Scalar-to-array equivalence:** Uniform per-box environment arrays produce the
  same results as scalar temperature/pressure for deterministic condensation and
  within accepted stochastic tolerances for coagulation.
- **Non-uniform environment execution:** Multi-box inputs with different
  temperature/pressure per box execute through the new path and expose per-box
  values to kernels.
- **Mismatch errors:** Wrong environment shapes and `environment.n_boxes` values
  raise `ValueError` before kernel launch.
- **Device errors:** Environment arrays on a different Warp device raise a clear
  validation error.
- **Ambiguity errors or precedence:** Passing both scalar values and explicit
  environment follows the documented contract and is tested.

## Test Locations

- Condensation behavior and validation:
  `particula/gpu/kernels/tests/condensation_test.py`.
- Coagulation behavior and validation:
  `particula/gpu/kernels/tests/coagulation_test.py`.
- Container/conversion behavior, if new environment container pieces are touched:
  `particula/gpu/tests/warp_types_test.py` and
  `particula/gpu/tests/conversion_test.py`.

## Verification Commands

- `pytest particula/gpu/kernels/tests/condensation_test.py -q`
- `pytest particula/gpu/kernels/tests/coagulation_test.py -q`
- `pytest particula/gpu/tests -q` when environment container or conversion code
  changes.
- `ruff check particula/ --fix && ruff format particula/ && ruff check particula/`
