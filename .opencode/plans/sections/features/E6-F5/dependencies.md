# Dependencies

## Upstream

- No E6 child feature is a prerequisite; E6-F5 may run in parallel with E6-F1
  and E6-F3.
- Shipped E5 particle containers, explicit conversion helpers, direct GPU
  preflight patterns, inactive-slot behavior, and Warp validation conventions
  are required infrastructure.
- `ParticleData` and `WarpParticleData` fixed-shape schemas are authoritative.
- Runtime dependency: NumPy on CPU and NVIDIA Warp for direct GPU execution;
  Warp CPU is required evidence and CUDA is optional.

## Downstream

- **E6-F6 / T6** requires exact free capacity and deterministic indices before
  choosing resampling or representative-volume scaling on exhaustion.
- **E6-F7 / T7** requires E6-F5 and E6-F6 to activate CPU nucleation source
  records only after gas-inventory and capacity policy finalization.
- **E6-F8 / T8** requires E6-F5, E6-F6, and E6-F7 for direct Warp nucleation.
- **E6-F9 / T9** requires E6-F1 through E6-F8 and validates integrated slot
  diagnostics and creation behavior.
- Epic G may later schedule these direct primitives but does not change their
  fixed-shape ownership contract.

## Sibling Boundaries

- E6-F3/E6-F4 may deactivate slots by clearing state but must use the same
  active/free semantics when interacting with E6-F5 diagnostics.
- E6-F5 reports insufficient capacity and fails atomically; it does not select
  or execute an E6-F6 exhaustion policy.
- E6-F7/E6-F8 own physical source masses and conservation, not slot ordering.

## Phase Ordering

P1 freezes CPU predicates and diagnostics before P2 CPU activation. P3 ports
read-only discovery to Warp before P4 adds device mutation and parity. P5 is the
required final documentation phase. Tests are co-located with every code phase.
