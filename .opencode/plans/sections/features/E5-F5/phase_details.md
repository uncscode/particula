# Phase Details

- [x] **E5-F5-P1:** Port ST1956 pair physics and kinematic-viscosity helpers with unit tests
  - Issue: #1352 | Size: S | Status: Completed
  - Delivered: Internal typed fp64 `kinematic_viscosity_wp()` and
    `turbulent_shear_st1956_pair_rate_wp()` helpers matching `nu = mu/rho` and
    ST1956. The rate helper returns exact zero before its cubic expression when
    dissipation is zero, avoiding `0 * inf` for finite extreme radii.
  - Files: `particula/gpu/dynamics/coagulation_funcs.py`, `particula/gpu/dynamics/tests/coagulation_funcs_test.py`
  - Tests: Independent fp64 NumPy parity and composed Sutherland-transport
    coverage; radius symmetry, cubic scaling, finite non-negative ordinary
    rates, and ordinary/overflow-guard exact-zero dissipation lanes.
  - Boundary: No public exports, APIs, dispatch, samplers, containers, or CPU
    fallbacks changed; P2 retains public-input validation.

- [x] **E5-F5-P2:** Validate explicit per-box dissipation and fluid-density inputs with unit tests
  - Issue: #1353 | Size: S | Status: Completed
  - Delivered: Added keyword-only `turbulent_dissipation` and `fluid_density`
    inputs plus `_ensure_turbulent_input_array` to the direct step. Enabled
    ST1956 requests accept positive finite floating scalars or supported
    same-device Warp `(n_boxes,)` arrays; supplied arrays retain identity and
    scalar broadcasts use private device storage. Valid P2 input reaches the
    unchanged reserved-capability error; no rate dispatch or execution shipped.
  - Files: `particula/gpu/kernels/coagulation.py`,
    `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: Helper scalar/array, dtype, shape, device, and invalid-category
    coverage; public ordering and failure-atomicity snapshots; runtime-helper
    sentinels; and non-turbulent ignored-argument regression coverage.

- [ ] **E5-F5-P3:** Integrate turbulent-shear-only sampling and safe majorant with execution tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Register turbulent shear, compute a proven per-box majorant, use one bounded acceptance pass, and apply accepted merges once.
  - Files: `particula/gpu/kernels/coagulation.py`, `particula/gpu/kernels/tests/coagulation_test.py`
  - Tests: All-pairs majorant proof, one/multi-box execution, inactive slots, deterministic pair rates, bounded stochastic behavior, mass conservation, buffer identity, persistent RNG reuse/reset, required Warp CPU, and optional CUDA.

- [ ] **E5-F5-P4:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document the direct call, input units/shape/device contract, supported ST1956 claim, no-DNS boundary, and E5-F6/F7 handoff.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`, `docs/Features/Roadmap/data-oriented-gpu.md`, `.opencode/plans/sections/features/E5-F5/*.md`
  - Tests: Markdown links, API names, examples, support-table language, and explicit no-DNS wording.
