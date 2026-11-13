# Infrastructure Reuse

- `coagulation_step_gpu()` in `particula/gpu/kernels/coagulation.py` is the
  public low-level orchestration boundary. P3 will extend it with a keyword-only
  configuration while preserving positional Brownian compatibility.
- Its validation ordering checks particle arrays, device, timestep, environment,
  volume, caller-owned buffers, and RNG state before allocation and launch. P3
  must insert mechanism preflight before normalization/allocation and retain this
  ordering.
- `brownian_coagulation_kernel` in `particula/gpu/kernels/coagulation.py`
  provides bounded active-pair selection, additive pair-rate/majorant dispatch,
  one acceptance draw per valid candidate, swap-pop removal, and persistent RNG
  patterns. Downstream terms must extend this loop rather than clone it.
- `brownian_kernel_pair_wp()` in
  `particula/gpu/dynamics/coagulation_funcs.py:92-139` is the first term behind
  the additive pair-rate interface and remains the independent Brownian
  implementation.
- `CombineCoagulationStrategy.kernel()` in
  `particula/dynamics/coagulation/coagulation_strategy/combine_coagulation_strategy.py:118-154`
  establishes CPU additive semantics: the combined kernel is the sum of each
  enabled strategy's kernel. Reuse the semantic rule, not CPU expected-value
  code, in Warp tests.
- `ThermodynamicsConfig` and its validator in
  `particula/gpu/kernels/thermodynamics.py:39-110` provide the preferred frozen
  sidecar, stable mode-code, metadata-validation, and concrete-module API
  pattern.
- `CondensationActivitySurfaceConfig` in
  `particula/gpu/kernels/condensation.py:90-128` demonstrates keyword-only
  configuration whose frozen bindings do not imply hidden ownership of device
  arrays.
- `WarpParticleData` in `particula/gpu/warp_types.py:24-78` supplies fixed-shape
  masses, concentration, charge, density, and volume. Do not add a distribution
  tag to this shared data schema; keep the support declaration in coagulation
  configuration.
- Existing regression and device fixtures in
  `particula/gpu/kernels/tests/coagulation_test.py:1-180` cover Warp import
  guards, CPU/CUDA parametrization, snapshots, persistent RNG, and direct-step
  behavior. Extend this module using the `*_test.py` convention.
- Follow `.opencode/guides/testing_guide.md:166-245` for required Warp CPU,
  optional CUDA, deterministic tolerance, and stochastic-bound policies.
