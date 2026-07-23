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

- **E6-F9 / T9** consumes the charged direct step in deterministic integrated
  validation and explicit-transfer documentation; E6-F4 retains charged
  statistical evidence for Epic F closeout.
- Epic G may later select/schedule the direct operation, but E6-F4 does not add
  backend selection, a resident loop, or a high-level runnable.

## Sibling Boundaries

- E6-F1/F2 dilution and E6-F7/F8 nucleation do not share wall-loss coefficient
  or RNG state.
- E6-F5/F6 own activation, diagnostics, resampling, representative-volume
  scaling, and exhaustion. E6-F4 only clears slots removed by wall loss.
- E6-F4 must not alter E6-F3's neutral results, public contract, or ownership.

## Phase Ordering

The shipped E6-F3 foundation preceded E6-F4 implementation. Within E6-F4, P1
froze semantics and validation before P2/P3 ported image and field physics. P4
integrated those terms into E6-F3's mutation path. P5 validated deterministic
coefficients, neutral fallback, stochastic outcomes, and RNG invariants. P6
shipped as #1414's documentation-only closeout, preserving E6-F3 as the
upstream foundation and E6-F9 as the downstream direct-call and
explicit-transfer consumer. Every implementation phase carries co-located
tests; there is no standalone testing phase.
