# Testing Strategy

Every implementation phase ships tests in the same change. Coverage thresholds
are never lowered; changed code maintains at least 80% coverage. GPU tests use
the repository's `*_test.py` convention and collect cleanly without Warp.

## Per-Phase Coverage

- **P1 -- physics helpers (completed):**
  `particula/gpu/dynamics/tests/coagulation_funcs_test.py` probes the internal
  fp64 kinematic-viscosity and ST1956 helpers directly and through composition
  with `dynamic_viscosity_wp`. Independent NumPy oracles cover transport,
  radius, dissipation, and density scales at `rtol=1e-12, atol=0.0`; direct
  rate tests verify radius symmetry, cubic scaling, finite non-negative
  ordinary results, and exact finite zero for ordinary and overflow-guard
  zero-dissipation lanes.
- **P2 -- input contract (completed):**
  `particula/gpu/kernels/tests/coagulation_test.py` covers Python/NumPy floating
  scalar broadcasts, heterogeneous active-device `wp.float64` `(n_boxes,)`
  arrays, and supplied-array identity. It rejects `wp.float32` arrays, missing,
  bool/integer, non-floating host/NumPy values, zero, negative, NaN/Inf,
  wrong-shape, unsupported-dtype, and wrong-device inputs. Public-step cases
  prove named P2 errors occur before mixed-turbulent capability rejection and
  snapshot particle mass, concentration, charge, collision buffers/counts, and
  RNG state for failure-atomicity. Sentinels prove invalid P2 values bypass
  downstream allocation, normalization, and RNG initialization or launches.
  Non-turbulent calls ignore supplied turbulence values.
- **P3 -- majorant and execution (completed):** Focused Warp tests enumerate all active unordered pairs and
  prove every independent ST1956 rate is finite, non-negative, and no greater
  than the device majorant. Cover zero/one/two active slots, inactive gaps,
  equal/different radii, one-box and heterogeneous multi-box state, bounded
  trial scheduling, sorted/in-range/disjoint pairs, mass conservation, output
  buffer identity, and persistent RNG reuse/reset. Use repeated-run means or
   sigma bounds for stochastic rates, never exact CPU/Warp pair replay. They
   also verify P2 failures and valid mixed turbulent-mask rejections occur
   before caller-owned particle, output, or RNG state is changed.
- **P4 -- documentation (completed):** Review the two feature pages and E5-F5
  plan records for direct imports/API names, SI units, scalar or active-device
  `wp.float64` `(n_boxes,)` inputs, caller ownership, ST1956-only support,
  no-DNS/additive boundaries, and E5-F6/E5-F7/E5-F9 handoffs. Run the focused
  `coagulation_test.py -q -Werror` Warp CPU baseline; CUDA is optional and
  skips cleanly when unavailable. No coverage or test-module change is needed.

## Device and Numerical Policy

- Warp CPU is required when Warp is installed and is the release baseline.
- CUDA is parametrized when available and skips cleanly otherwise.
- Deterministic fixtures use explicit `np.float64` values and declared
  scale-appropriate `rtol`/`atol`; conservation stays tight per box/species.
- Expected values come from the public CPU ST1956 function or direct NumPy
  equations and do not call the new Warp helper.
- No test imports or executes turbulent DNS implementations as evidence for
  this feature.

## Coverage Impact

Primary orchestration coverage remains colocated in
`particula/gpu/kernels/tests/coagulation_test.py`. A focused
`particula/gpu/kernels/tests/turbulent_shear_coagulation_test.py` may be created
if clarity requires it. Required correctness tests need no slow/performance
marker; E5-F5 makes no throughput claim.
