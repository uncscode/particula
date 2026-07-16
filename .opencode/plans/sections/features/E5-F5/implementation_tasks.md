# Implementation Tasks

## GPU Physics

- [ ] Add a typed fp64 ST1956 pair function in
  `particula/gpu/dynamics/coagulation_funcs.py` with radius, dissipation, and
  kinematic-viscosity arguments and documented SI units.
- [ ] Reuse `dynamic_viscosity_wp()` and add or inline the reviewed
  `kinematic_viscosity = dynamic_viscosity / fluid_density` operation.
- [ ] Add independent helper probes and CPU/NumPy equation comparisons in
  `particula/gpu/dynamics/tests/coagulation_funcs_test.py`.

## Coagulation Orchestration

- [ ] Add keyword-only `turbulent_dissipation` and `fluid_density` inputs to
  the E5-F1 concrete direct-step contract without changing legacy Brownian
  positional calls.
- [ ] Implement one reusable scalar-or-Warp `(n_boxes,)` positive-finite input
  normalizer in `particula/gpu/kernels/coagulation.py`; preserve valid caller
  arrays by identity and reject NumPy arrays/unsupported dtypes/devices.
- [ ] Require both inputs only for configurations containing turbulent shear
  and apply E5-F1's policy to irrelevant supplied inputs.
- [ ] Derive the per-box ST1956 prefactor and active radii entirely on device.
- [ ] Implement and document the safe majorant; add the turbulent term to the
  shared pair-rate/majorant dispatcher and capability matrix.
- [ ] Route turbulent-shear-only execution through the existing bounded
  candidate stream, one acceptance draw, one merge launch, caller buffers, and
  caller-owned persistent RNG state.
- [ ] Guard non-finite/negative device results and preserve the defensive
  acceptance-ratio clamp without masking caller validation errors.

## Tooling / Tests

- [ ] Add all-pairs pair-rate and majorant checks to
  `particula/gpu/kernels/tests/coagulation_test.py` or a focused
  `turbulent_shear_coagulation_test.py` if module size requires it.
- [ ] Add scalar/per-box, heterogeneous multi-box, inactive-slot, zero/one/two
  active, buffer identity, RNG reuse/reset, stochastic, and conservation cases.
- [ ] Snapshot masses, concentration, charge, output buffers, and RNG state for
  every invalid/unsupported preflight case.
- [ ] Run required Warp CPU tests and parametrized optional CUDA tests; use
  explicit fp64 tolerances and aggregate/sigma stochastic assertions.

## Documentation

- [ ] Document import path, inputs, units, shape/device rules, and explicit
  ST1956-only/no-DNS support language.
- [ ] Update E5 handoff references for E5-F6, E5-F7, and E5-F9 without claiming
  additive combinations before those tracks ship.
