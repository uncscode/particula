# Dependencies

## Upstream

- **E6-F3 / T3 is mandatory:** E6-F4 extends its neutral spherical/rectangular
  coefficients, immutable configuration, environment normalization, fixed-slot
  removal, preflight ordering, lazy export, and persistent RNG lifecycle.
- Existing CPU `ChargedWallLossStrategy` is the authoritative image-charge,
  field, potential, sign, clipping, and neutral-fallback oracle; this feature
  must not redefine CPU semantics.
- Shipped E5 coagulation supplies caller-owned charge, persistent RNG,
  deactivation, Warp CPU/optional CUDA, and no-hidden-transfer conventions.
- Runtime dependency: NVIDIA Warp. CUDA hardware is optional and non-blocking.

## Downstream

- **E6-F9 / T9** consumes the charged direct step in integrated validation,
  explicit-transfer documentation, and Epic F closeout.
- Epic G may later select/schedule the direct operation, but E6-F4 does not add
  backend selection, a resident loop, or a high-level runnable.

## Sibling Boundaries

- E6-F1/F2 dilution and E6-F7/F8 nucleation do not share wall-loss coefficient
  or RNG state.
- E6-F5/F6 own activation, diagnostics, resampling, representative-volume
  scaling, and exhaustion. E6-F4 only clears slots removed by wall loss.
- E6-F4 must not alter E6-F3's neutral results, public contract, or ownership.

## Phase Ordering

E6-F3 must ship before E6-F4 implementation. Within E6-F4, P1 freezes semantics
and validation before P2/P3 port image and field physics. P4 integrates those
terms into E6-F3's mutation path. P5 validates deterministic coefficients,
neutral fallback, stochastic outcomes, and RNG invariants. P6 is the required
final documentation phase. Every implementation phase carries co-located tests;
there is no standalone testing phase.
