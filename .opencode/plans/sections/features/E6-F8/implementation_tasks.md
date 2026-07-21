# Implementation Tasks

## GPU Backend

- [ ] Define the bounded config, scratch, diagnostic, and request sidecars in
  `particula/gpu/kernels/nucleation.py`, documenting every dtype and shape.
- [ ] Implement metadata, alias, same-device, scientific-domain, count, and
  read-only device-value preflight before clearing or allocating outputs.
- [ ] Port E6-F7 activation/kinetic rate equations and SI conversions to Warp
  kernels without broadening their validity domain or model aliases.
- [ ] Implement potential-event and shared per-box gas-inventory finalization;
  store admitted events, limiting species, and gas removal explicitly.
- [ ] Adapt finalized demand to E6-F5 fixed-shape requests and exact diagnostics
  rather than reimplementing active/free predicates.
- [ ] Invoke E6-F6 complete-demand exhaustion planning with resampling-first and
  scaling-fallback precedence; reject any residual or truncated demand.
- [ ] Implement commit kernels that add represented particle source and subtract
  the exact finalized gas mass only after every box is feasible.
- [ ] Add `nucleation_step_gpu(...)` with stable return/identity behavior and
  the intended lazy export in `particula/gpu/kernels/__init__.py`.
- [ ] Confirm the step contains no conversion-helper call, `.numpy()` physics
  path, CPU fallback, dynamic resize, or implicit synchronization contract.

## Tooling / Tests

- [ ] Add fast validation, rate, finalization, capacity, identity, no-op, and
  failure-atomicity tests in `particula/gpu/kernels/tests/nucleation_test.py`.
- [ ] Add `nucleation_parity_test.py` with an independent float64 E6-F7 oracle,
  one/many boxes and species, sparse/full slots, and repeated calls.
- [ ] Assert per-box/species represented particle-plus-gas conservation rather
  than aggregate conservation alone.
- [ ] Require Warp CPU when Warp is installed and make CUDA parametrization skip
  cleanly when unavailable; retain at least 80% changed-code coverage.
- [ ] Add regression assertions for explicit transfer boundaries, stable shapes,
  dtypes/devices/identities, and no mutation after every rejected call.
