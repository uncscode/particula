# Testing Strategy

Unit tests ship with every phase. Use explicit fp64 fixtures and compare against
CPU inventory-limit semantics. Required cases include condensation with ample
and insufficient gas, evaporation, mixed positive/negative transfer, disabled
species, inactive and zero-concentration slots, and independent multi-box and
multi-species limits.

Conservation is asserted independently per box/species: concentration-weighted
particle gain equals gas loss to tight bookkeeping tolerance; both inventories
remain finite and nonnegative. Global sums are supplemental only. Exactly four
substeps must use updated gas, and caller-owned buffers must retain identity and
validate before mutation. Returned transfer and E4-F4 energy must use the exact
applied transfer.

The issue #1272 production hook and regression in
`particula/integration_tests/condensation_particle_resolved_test.py` land in the
same phase. Warp CPU is required when Warp is installed; CUDA runs the same
checks when available and otherwise skips cleanly. Physics parity tolerances are
reported separately from stricter conservation tolerances. Tests must detect
hidden host conversion/synchronization and preserve the public API contract.
