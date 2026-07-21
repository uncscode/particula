# Dependencies

## Upstream

- **E6-F5 / T5 (Particle Slot Discovery and Activation):** mandatory source of
  fixed-shape active/free diagnostics, deterministic free indices, request
  contracts, and atomic activation. Plan metadata must retain this dependency.
- **E6-F6 / T6 (Particle Slot Exhaustion Handling):** mandatory source of
  complete-demand resampling-first planning and optional representative-volume
  scaling fallback. Plan metadata must retain this dependency.
- **E6-F7 / T7 (CPU Nucleation and Particle-Source Process):** mandatory
  scientific, inventory-finalization, no-op, mutation, and float64 oracle
  contract. GPU work must not begin by inventing alternate physics.
- Warp GPU containers, explicit conversion helpers, and direct condensation
  finalization patterns are repository infrastructure, not fallback paths.

## Downstream

- **E6-F9 / T9** consumes this direct step with E6's other low-level processes
  in an explicit-transfer integrated example and compatibility matrix.
- Future Epic G scheduling/backend selection may orchestrate this step but may
  not retroactively add hidden transfer or fallback to the E6-F8 contract.

## Phase Ordering

P1 freezes sidecars and validation before P2 writes finalization kernels. P3
requires E6-F5; P4 requires E6-F6 and finalized demand from P2. P5 composes P2
through P4 only after all contracts are tested. P6 supplies integration parity
and conservation evidence before final P7 documentation. E6-F5, E6-F6, and
E6-F7 remain dependencies of the whole feature, not merely optional phases.
