# Open Questions

All E6-F4 planning questions were resolved on 2026-07-21 against the current
CPU charged-wall-loss semantics and direct-kernel testing conventions.

- [x] Does nonzero charge retain image enhancement at zero wall potential?
  - Decision: yes. Wall potential does not gate the self-image factor.
- [x] Does configured field or potential alter a zero-charge particle?
  - Decision: no. It takes the shared E6-F3 neutral branch.
- [x] Does charged mode introduce a separate step API or RNG stream?
  - Decision: no. It extends E6-F3 configuration and reuses its direct step,
    active predicate, clearing, and caller-owned RNG sidecar.
- [x] Which rectangular electric-field forms are accepted?
  - Decision: accept only the public CPU builder's finite three-component vector
    form and resolve it by Euclidean norm. Scalar fields remain spherical-only;
    their finite signed value preserves current CPU strategy behavior.
    Permissive rectangular scalar behavior does not expand the GPU contract.
- [x] How strong is zero-charge fallback parity?
  - Decision: require bitwise-identical same-device neutral coefficients,
    survivor state, and final RNG state when streams begin identically. Branch
    directly to E6-F3 before charged arithmetic.
- [x] Which deterministic and statistical bounds cover the charged matrix?
  - Decision: test 2 nm, 50 nm, 1 micrometer, and 50 micrometer strata. Full
    CPU/Warp coefficients use `rtol=1e-6`, `atol=0`; reused/component helpers
    retain their tighter established tolerances and exact branches remain exact.
    Each homogeneous survival stratum uses 4,096 observations. The eight charged
    strata are four radii crossed with image-only spherical and
    field-plus-potential rectangular cases. Use an exact binomial interval with
    family-wise alpha `1e-6`, hence per-stratum alpha `1.25e-7`.
