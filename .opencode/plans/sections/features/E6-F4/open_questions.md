# Open Questions

P1 configuration questions shipped for issue #1409, P2 private primitive
questions shipped for issue #1410, and P3 field/drift/composition questions
 shipped for issue #1411 on 2026-07-23. Issue #1412 implemented P4 direct-step
 selection, and issue #1413 shipped P5 regression evidence only.

- [x] What image-charge primitive contract is available before step integration?
  - Decision: P2 provides private fp64 Warp self-pair Coulomb-ratio and
    enhancement helpers. The ratio lower-clips at `-200`; enhancement uses the
    absolute ratio, `[-50, 50]` exponent clamp, and exact zero-charge identity.
    Independent NumPy/Warp parity and clipping tests cover it. No helper is
    exported; P4 calls it privately from the direct wall-loss kernel.

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
    charged-spherical-only. P4 resolves the validated forms only in their
    respective charged kernels.
- [x] How strong is zero-charge fallback parity?
  - Decision: P4 preserves the neutral branch for exact zero charge, so matched
    zero-charge charged and neutral calls have exact particle and final-RNG
    equality.
- [x] How are geometry, electric fields, drift, and unsafe coefficients handled
  before direct-step integration?
  - Decision: P3 privately selects radius or minimum rectangular dimension;
    resolves signed spherical fields and rectangular vector norms with the CPU
    conditional potential contribution; calculates signed mobility drift with
     radius/scale guards; and clips the composed rate to finite nonnegative fp64.
    The helpers do not validate configuration or mutate state; P4 calls them
    only from nonzero-charge geometry-specialized kernels.
- [x] How is charged selection integrated without changing the direct boundary?
  - Decision: P4 dispatches one existing neutral or one geometry-specialized
    charged removal kernel after frozen preflight and the zero-time return.
    Neutral mode is unchanged; exact zero-charge charged slots take its
    coefficient/RNG path. The rectangular vector remains caller-owned and is
    passed only to charged rectangular execution.
- [x] How do charged composed-rate edge cases affect RNG ownership?
  - Decision: nonpositive composed rates do not draw. Charged saturation uses
    the standard survival draw before underflow-probability removal, rather than
    the neutral positive-infinity no-draw shortcut. Invalid, zero-time, and
    all-inactive calls leave supplied sidecars untouched.
- [x] Which deterministic and statistical bounds cover the charged matrix?
  - Decision: #1413 tests 2 nm, 50 nm, 1 micrometer, and 50 micrometer strata.
    Charged CPU/Warp parity uses spherical `rtol=1.002e-3, atol=1e-20` because
    of the established Debye integration difference, and rectangular
    `rtol=1e-6, atol=0`. Each homogeneous stratum uses 4,096 observations; the
    eight strata cross four radii with image-only spherical and
    field-plus-potential rectangular cases. Inclusive equal-tail exact-binomial
    intervals use family-wise alpha `1e-6`, hence per-stratum alpha `1.25e-7`.
