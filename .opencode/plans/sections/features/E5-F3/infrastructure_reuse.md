# Infrastructure Reuse

- `coagulation_step_gpu` in `particula/gpu/kernels/coagulation.py:1383` is the
  low-level orchestration boundary; extend its E5-F1 keyword-only mechanism
  configuration rather than adding another public step.
- The bounded active-pair loop in
  `particula/gpu/kernels/coagulation.py:641-963` already schedules trials,
  selects disjoint active pairs, accepts once, and advances one RNG stream.
- `_bound_scheduled_trials`, `_select_active_pair_by_rank`, and
  `_remove_active_pair_by_rank_swap_pop` in
  `particula/gpu/kernels/coagulation.py:274-282 and 536-574` preserve trial
  limits and disjoint-pair semantics.
- Brownian property preparation and `brownian_kernel_pair_wp` use the focused
  scalar-function pattern in `particula/gpu/dynamics/coagulation_funcs.py:10-158`.
  E5-F2 charged helpers should be called from the same device-side dispatcher.
- The current Brownian exact active-pair majorant is implemented at
  `particula/gpu/kernels/coagulation.py:822-856`; retain it as the Brownian term
  and add the charged term's independently safe bound.
- `apply_coagulation_kernel` in `particula/gpu/kernels/coagulation.py:969-1013`
  is extended by E5-F2 to transfer and clear charge. Reuse it once after the
  shared pair-selection pass.
- Validation and ownership helpers at
  `particula/gpu/kernels/coagulation.py:1016-1380` provide shape/device/domain,
  output-buffer, volume, and persistent-RNG preflight patterns.
- `particula/gpu/kernels/tests/coagulation_test.py:267-465` provides particle
  fixtures, state snapshots, seeded step helpers, and supported-device
  parameterization for integration and stochastic tests.
- `particula/gpu/dynamics/tests/coagulation_funcs_test.py` demonstrates
  independent NumPy/Warp scalar parity probes; follow it for deterministic
  majorant and rate checks without using GPU helpers as the oracle.
- CPU `CombineCoagulationStrategy` and charged strategy/formula modules under
  `particula/dynamics/coagulation/` define independent additive semantics and
  expected physics, not runtime dependencies of the GPU path.
