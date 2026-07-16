# Infrastructure Reuse

- `coagulation_step_gpu` in `particula/gpu/kernels/coagulation.py:792` is the
  low-level orchestration boundary; extend its E5-F1 keyword-only mechanism
  configuration rather than adding another public step.
- The bounded active-pair loop in
  `particula/gpu/kernels/coagulation.py:345-427` already schedules trials,
  selects disjoint active pairs, accepts once, and advances one RNG stream.
- `_bound_scheduled_trials`, `_select_active_pair_by_rank`, and
  `_remove_active_pair_by_rank_swap_pop` in
  `particula/gpu/kernels/coagulation.py:66-119` preserve trial limits and
  disjoint-pair semantics.
- Brownian property preparation and `brownian_kernel_pair_wp` use the focused
  scalar-function pattern in `particula/gpu/dynamics/coagulation_funcs.py:10-139`.
  E5-F2 charged helpers should be called from the same device-side dispatcher.
- The current Brownian extreme-radius majorant is implemented at
  `particula/gpu/kernels/coagulation.py:300-343`; retain it as the Brownian term
  and add the charged term's independently safe bound.
- `apply_coagulation_kernel` in `particula/gpu/kernels/coagulation.py:430-465`
  is extended by E5-F2 to transfer and clear charge. Reuse it once after the
  shared pair-selection pass.
- Validation and ownership helpers at
  `particula/gpu/kernels/coagulation.py:468-789` provide shape/device/domain,
  output-buffer, volume, and persistent-RNG preflight patterns.
- `particula/gpu/kernels/tests/coagulation_test.py:269-459` provides particle
  fixtures, state snapshots, seeded step helpers, and supported-device
  parameterization for integration and stochastic tests.
- `particula/gpu/dynamics/tests/coagulation_funcs_test.py` demonstrates
  independent NumPy/Warp scalar parity probes; follow it for deterministic
  majorant and rate checks without using GPU helpers as the oracle.
- CPU `CombineCoagulationStrategy` and charged strategy/formula modules under
  `particula/dynamics/coagulation/` define independent additive semantics and
  expected physics, not runtime dependencies of the GPU path.
