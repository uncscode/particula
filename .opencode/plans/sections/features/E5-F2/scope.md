# Scope

E5-F2 ports the scalar Coulomb, reduced-property, and approved charged
pair-kernel calculations needed by the E5 mechanism dispatcher; validates the
existing fp64 charge field; and extends accepted-pair application to add donor
charge into the recipient before clearing the donor. It does not make charged
 physics executable through the public sampler; that integration belongs to
E5-F3.

**Completed increment (E5-F2-P1, issue #1336):** internal scalar fp64 Warp
Coulomb/reduced-property helpers and independent co-located parity tests are
implemented only in `particula/gpu/dynamics/coagulation_funcs.py` and
`particula/gpu/dynamics/tests/coagulation_funcs_test.py`. This increment does
not change public exports, data containers, Brownian dispatch, charged
execution wiring, or module boundaries.

**Completed increment (E5-F2-P2, issue #1337):** the same two files now add
the internal scalar `charged_hard_sphere_wp` helper and its independent NumPy
oracle/probe coverage. It composes the existing property and Coulomb helpers,
returns exact safe zero for invalid, non-finite, or non-positive physical
inputs, and has no execution or dispatch integration.

## In Scope

- Add focused `@wp.func` helpers in
  `particula/gpu/dynamics/coagulation_funcs.py` for approved charged pair
  physics, including stable neutral, attractive, and repulsive limits.
- Use CPU formulas as independent references while keeping device execution
  free of hidden host calculations or transfers.
- Validate `WarpParticleData.charge` shape, dtype, device, and finite values
  before particle mutation or RNG advancement.
- Pass charge into `apply_coagulation_kernel`, add donor charge to the accepted
  recipient, and clear donor charge with donor mass and concentration.
- Add co-located deterministic pair parity, validation, merge, multi-box,
  inactive-slot, and separate mass/charge conservation tests.
- Preserve Brownian behavior for all-zero charge and existing callers.

## Out of Scope

- Candidate selection, charged majorant proofs, stochastic charged execution,
  or Brownian-plus-charged execution (E5-F3).
- Sedimentation, turbulent shear, and additive multi-mechanism execution
  (E5-F4 through E5-F6).
- Cross-mechanism release matrix and user-facing support closeout (E5-F7/F9).
- Binned or continuous-PDF GPU coagulation, high-level `Aerosol`/`Runnable`
  APIs, hidden CPU fallback/transfers, dynamic slots, graph capture, autodiff,
  adaptive stepping, DNS turbulence, or general performance redesign.
