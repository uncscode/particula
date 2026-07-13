# Dependencies

## Upstream

- **E4-F1:** On-device thermodynamic/vapor-pressure refresh and validation.
- **E4-F2:** Activity and Kelvin physics with explicit supported modes.
- **E4-F3:** Exactly four substeps and reusable fixed-shape scratch.
- **E4-F4:** Thermal feedback and latent-energy bookkeeping sidecars.
- **E4-F5:** Gas coupling, bounded transfer, partitioning gates, and strict
  particle-plus-gas inventory semantics.

All five upstream tracks must expose their final production API before parity
fixtures and acceptance tolerances are frozen. Parent epic **E4** orders E4-F6
after E4-F5.

## Downstream

- **E4-F7** consumes this qualification evidence and must not imply stronger
  graph/autodiff support than E4-F6 demonstrates.
- Issue 1272 closure and user-facing support claims depend on this matrix.

## Phase Ordering

P1 establishes reference parity; P2 adds invariants and mutation contracts.
P3 uses those oracles for graph replay. P4 evaluates bounded autodiff readiness
after access and clamp boundaries are known. P5 documents only verified results.
