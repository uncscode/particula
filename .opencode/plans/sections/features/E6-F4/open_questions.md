# Open Questions

P1 configuration questions shipped for issue #1409, P2 private primitive
questions shipped for issue #1410, and P3 field/drift/composition questions
shipped for issue #1411 on 2026-07-23. Direct-step integration and validation
questions remain deferred to P4-P5.

- [x] What image-charge primitive contract is available before step integration?
  - Decision: P2 provides private fp64 Warp self-pair Coulomb-ratio and
    enhancement helpers. The ratio lower-clips at `-200`; enhancement uses the
    absolute ratio, `[-50, 50]` exponent clamp, and exact zero-charge identity.
    Independent NumPy/Warp parity and clipping tests cover it. No helper is
    exported or called by the direct wall-loss kernel.

- [x] Does nonzero charge retain image enhancement at zero wall potential?
  - Decision: yes. Wall potential does not gate the self-image factor.
- [x] Does configured field or potential alter a zero-charge particle?
  - Decision: no. It takes the shared E6-F3 neutral branch.
- [x] Does charged mode introduce a separate step API or RNG stream?
  - Decision: no. It extends E6-F3 configuration and reuses its direct step,
    active predicate, clearing, and caller-owned RNG sidecar.
- [x] Which rectangular electric-field forms are accepted?
  - Decision: P1 accepts only a caller-owned, same-device finite `wp.float64`
    `(3,)` Warp vector in charged rectangular mode. Scalar fields are
    charged-spherical-only. P1 validates but does not resolve either form for
    execution.
- [x] How strong is zero-charge fallback parity?
  - Decision: P1 leaves all execution neutral, so matched zero-charge charged
     and neutral calls have exact particle and final-RNG equality. Future
     charged arithmetic must preserve this fallback.
- [x] How are geometry, electric fields, drift, and unsafe coefficients handled
  before direct-step integration?
  - Decision: P3 privately selects radius or minimum rectangular dimension;
    resolves signed spherical fields and rectangular vector norms with the CPU
    conditional potential contribution; calculates signed mobility drift with
    radius/scale guards; and clips the composed rate to finite nonnegative fp64.
    The helpers do not validate configuration, mutate state, or have a kernel
    call site; P4 owns that integration.
- [x] Which deterministic and statistical bounds cover the charged matrix?
  - Decision: test 2 nm, 50 nm, 1 micrometer, and 50 micrometer strata. Full
    CPU/Warp coefficients use `rtol=1e-6`, `atol=0`; reused/component helpers
    retain their tighter established tolerances and exact branches remain exact.
    Each homogeneous survival stratum uses 4,096 observations. The eight charged
    strata are four radii crossed with image-only spherical and
    field-plus-potential rectangular cases. Use an exact binomial interval with
    family-wise alpha `1e-6`, hence per-stratum alpha `1.25e-7`.
