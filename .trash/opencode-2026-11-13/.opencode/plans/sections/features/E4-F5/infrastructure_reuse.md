# Infrastructure Reuse

- Extend `particula/gpu/kernels/condensation.py` and retain the public
  `condensation_step_gpu` entry point and explicit conversion boundary.
- Reuse `WarpGasData.concentration`, `vapor_pressure`, and int32 `partitioning`
  from `particula/gpu/warp_types.py`; no schema addition is required.
- Port the separate positive/negative scaling semantics from
  `dynamics/condensation/mass_transfer_utils.py` and the coupled update order
  from `condensation_strategies.py`.
- Reuse E4-F3 fixed four-substep, stable-shape, caller-owned scratch patterns
  from `_condensation_test_support.py`.
- Drive E4-F4 thermal correction and energy accounting with the same finalized
  transfer accumulator.
- Extend `condensation_test.py`, `condensation_stiffness_test.py`, and
  `integration_tests/condensation_particle_resolved_test.py`; use latent-heat
  conservation tests as the bookkeeping pattern.
