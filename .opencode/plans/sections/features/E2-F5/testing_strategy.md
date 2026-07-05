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
- **Shared helper coverage:** `_ensure_environment_arrays(...)` broadcasts
  scalar inputs, returns valid direct/environment-backed arrays unchanged,
  accepts hybrid scalar-plus-array direct inputs, and rejects wrong shape,
  wrong device, mixed direct-plus-environment inputs, and missing direct
  inputs.
- **Explicit-environment execution:** Passing `temperature=None`,
  `pressure=None`, and valid `environment=...` succeeds for both GPU entry
  points.
- **Direct array execution:** Direct Warp-array temperature/pressure inputs and
  hybrid scalar-plus-array direct inputs succeed for both entry points.
- **Condensation regression hardening:** Condensation coverage proves scalar
  versus uniform `(n_boxes,)` equivalence, non-uniform explicit-environment
  parity, invalid temperature/pressure domain failures before launch, rejected
  input preservation of caller-owned `mass_transfer` buffers, and zero-volume
  particle short-circuit safety with per-box arrays.
- **Coagulation regression hardening:** Coagulation coverage proves uniform
  direct `(n_boxes,)` arrays match scalar behavior within established
  stochastic tolerance, non-uniform environment inputs change per-box collision
  trends in the CPU Brownian-rate direction, and boxes with fewer than two
  active particles still record zero collisions.
- **Pre-launch short-circuiting:** Contract failures occur before
  `get_dynamic_viscosity(...)`, `get_molecule_mean_free_path(...)`,
  `_ensure_volume_array(...)`, or any Warp launch.
- **Reuse/performance regressions:** Condensation prepares box properties once
  per call, and coagulation forwards validated direct arrays without rebuilding
  them.

## Test Locations

- Shared helper behavior:
  `particula/gpu/kernels/tests/environment_test.py`.
- Condensation behavior and validation:
  `particula/gpu/kernels/tests/condensation_test.py`.
- Coagulation behavior and validation:
  `particula/gpu/kernels/tests/coagulation_test.py`.
- Formatting-only companion updates for shared GPU test readability may appear
  in companion files; shipped P4 behavior is validated in
  `coagulation_test.py`, while `condensation_test.py` only saw formatting-only
  churn during the phase.
- Container/conversion behavior stays covered by existing helpers; this phase
  reuses `EnvironmentData` plus `to_warp_environment_data(...)` inside the
  co-located kernel/helper tests instead of changing conversion test scope.

## Verification Commands

- `pytest particula/gpu/kernels/tests/environment_test.py -q`
- `pytest particula/gpu/kernels/tests/condensation_test.py -q`
- `pytest particula/gpu/kernels/tests/coagulation_test.py -q`
- `ruff check particula/ --fix && ruff format particula/ && ruff check particula/`
