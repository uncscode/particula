# Infrastructure Reuse

- `coagulation_step_gpu()` and the bounded sampler in
  `particula/gpu/kernels/coagulation.py` are the system under test; exercise the
  public low-level boundary rather than constructing a second execution path.
- Existing Warp import guards, particle factories, snapshots, test-local
  sampler diagnostics, and device parametrization in
  `particula/gpu/kernels/tests/coagulation_test.py:1-180` and
  `:847-1019` provide reusable fixture and ownership patterns.
- Brownian matrix parity, multi-box independence, mass conservation, reusable
  buffers, and inactive-slot regressions already live at
  `particula/gpu/kernels/tests/coagulation_test.py:1884`, `:1975`, `:2562`,
  `:2654`, and `:3188`; generalize their assertions instead of duplicating
  isolated variants.
- `warp_devices()` and `CUDA_SKIP_REASON` in
  `particula/gpu/tests/cuda_availability.py:14-37` define the stable required-CPU
  and optional-CUDA test contract.
- `WarpParticleData` in `particula/gpu/warp_types.py:24-78` provides fixed-shape
  masses, concentration, charge, density, and volume for one- and multi-box
  fixtures.
- The physical-tolerance-floor pattern in
  `particula/gpu/kernels/tests/_condensation_test_support.py:316-329` can be
  adapted for per-box/per-species coagulation conservation without weakening
  small-mass checks.
- CPU `CombineCoagulationStrategy.kernel()` in
  `particula/dynamics/coagulation/coagulation_strategy/combine_coagulation_strategy.py:118-154`
  defines additive semantics. Use public CPU formulas or direct NumPy equations
  as independent expected values; never call the Warp aggregate helper as its
  own oracle.
- Mechanism-specific parity tests planned by E5-F3, E5-F4, E5-F5, and E5-F6
  supply reusable charged, SP2016, ST1956, and additive fixtures. E5-F7 owns
  cross-row consistency and fills only matrix gaps.
- `.opencode/guides/testing_guide.md:221-235` defines the separation of parity,
  conservation, and bounded stochastic criteria; `:310-313` defines aggregate
  stochastic bounds.
