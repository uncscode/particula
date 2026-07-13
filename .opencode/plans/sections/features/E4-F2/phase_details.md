# Phase Details

- [ ] **E4-F2-P1:** Ideal and kappa activity Warp helpers with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Implement ideal molar and kappa water activity with guarded fp64 formulas.
   - Files: `particula/gpu/dynamics/condensation_funcs.py` (new
     `water_activity_ideal_wp()` and `water_activity_kappa_wp()` helpers),
     `particula/gpu/dynamics/tests/condensation_funcs_test.py`
  - Tests: Pure/mixed, wet/dry, zero-solute, multi-solute CPU formula parity.
- [ ] **E4-F2-P2:** Static and composition-weighted surface physics with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Preserve static input and add the selected composition-weighted mode.
   - Files: `particula/gpu/dynamics/condensation_funcs.py` (new
     `effective_surface_tension_wp()` helper),
     `particula/gpu/dynamics/tests/condensation_funcs_test.py`
  - Tests: Static compatibility, mixtures, zero weights, Kelvin inputs.
- [ ] **E4-F2-P3:** Activity-adjusted Kelvin integration and validation tests
  - Issue: TBD | Size: S | Status: Not Started
   - Goal: Pass numeric configuration through `condensation_step_gpu()` and
     compose activity, E4-F1 pure pressure, and Kelvin pressure in its transfer
     launch (keep this production-path delta to roughly 100 LOC).
   - Files: `particula/gpu/kernels/condensation.py`,
     `particula/gpu/kernels/tests/condensation_test.py`
  - Tests: Launch integration, invalid configuration, failure-before-mutation.
- [ ] **E4-F2-P4:** CPU and optional CUDA parity fixtures plus documentation
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Verify coupled physics and document supported/deferred behavior.
  - Files: GPU condensation tests, `docs/Features/`, plan sections
  - Tests: Warp CPU required, CUDA optional skip, docs/link validation.
