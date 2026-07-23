# Open Questions

P1 configuration questions were resolved and shipped for issue #1409 on
2026-07-23. Charged-physics questions remain deferred to P2-P5.

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
- [x] Which deterministic and statistical bounds cover the charged matrix?
  - Decision: test 2 nm, 50 nm, 1 micrometer, and 50 micrometer strata. Full
    CPU/Warp coefficients use `rtol=1e-6`, `atol=0`; reused/component helpers
    retain their tighter established tolerances and exact branches remain exact.
    Each homogeneous survival stratum uses 4,096 observations. The eight charged
    strata are four radii crossed with image-only spherical and
    field-plus-potential rectangular cases. Use an exact binomial interval with
    family-wise alpha `1e-6`, hence per-stratum alpha `1.25e-7`.
