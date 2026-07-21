# Dependencies

## Upstream

- No E6 child feature blocks E6-F3; parent E6 explicitly permits E6-F1, E6-F3,
  and E6-F5 to begin in parallel.
- Shipped E5 direct coagulation supplies fixed-shape clearing, persistent RNG,
  validation, and Warp CPU/optional CUDA conventions.
- Existing direct condensation and Warp environment/container helpers supply
  explicit ownership, scalar/per-box normalization, and no-hidden-transfer
  patterns.
- Existing CPU `SphericalWallLossStrategy`, `RectangularWallLossStrategy`, and
  coefficient/property functions are mandatory behavioral oracles.
- Runtime dependency: NVIDIA Warp. CUDA hardware is optional and non-blocking.

## Downstream

- **E6-F4 / T4:** charged GPU wall loss extends this neutral configuration,
  transport, coefficient, removal, RNG, and fallback foundation.
- **E6-F9 / T9:** integrated validation and documentation consume this direct
  neutral step and persistent sidecar behavior.
- Epic G may later schedule this operation, but backend selection and resident
  loops remain outside E6-F3 and must not shape this API beyond explicit,
  low-level ownership.

## Sibling Boundaries

- E6-F1/F2 dilution do not share mutation or RNG state with wall loss.
- E6-F5/F6 own generic activation, diagnostics, exhaustion, and resampling;
  E6-F3 only deactivates removed slots according to the established fixed-slot
  clearing invariant and does not activate or compact slots.
- E6-F7/F8 nucleation are unrelated to neutral wall-loss physics.

## Phase Ordering

P1 establishes device transport primitives before P2 coefficients. P3 freezes
the public/configuration and atomic preflight contract before P4 mutation. P5
adds the reusable persistent-RNG lifecycle after the core step exists. P6 runs
the complete deterministic/statistical parity matrix and export smoke tests.
P7 is the required final development-documentation phase. Every implementation
phase ships co-located tests; there is no standalone testing phase.
