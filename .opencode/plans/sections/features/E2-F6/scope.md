# E2-F6 Scope

## In Scope

- Define deterministic numerical cases spanning NPF-scale particles,
  accumulation-mode particles, and droplet-scale particles.
- Establish the current absolute per-species `fp64` mass storage as the
  reference baseline.
- Compare candidate alternatives:
  - absolute `fp32` storage,
  - mixed precision such as `fp32` storage with `fp64` accumulation/reference
    calculations,
  - representation alternatives noted in the roadmap such as log-mass or
    reference/binned mass scaling.
- Measure conservation, small-particle radius/mass fidelity, nonnegative mass
  behavior, clamping frequency, memory budget, and throughput where practical.
- Produce a precision/mass representation report with validation evidence and a
  recommendation.

## Out of Scope

- Changing default `ParticleData` or `WarpParticleData` dtype.
- Replacing the canonical `(n_boxes, n_particles, n_species)` mass schema.
- Shipping production mixed-precision kernels.
- Reworking gas/environment ownership from E2-F2 through E2-F5.
- Treating current GPU condensation clamping as sufficient conservation proof
  without a reference comparison.

## Done Boundary

The feature is complete when the report exists, cites reproducible validation
cases, and gives a recommendation before any schema or dtype changes are made.
