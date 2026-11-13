# Dependencies

## Upstream

- Issue #1272 is the authoritative source for dependencies and acceptance
  criteria; this plan preserves its constant/Buck, on-device refresh, validation,
  current-temperature, CPU-parity, and fail-early requirements.
- Shipped E1/E2/E3 GPU container, conversion, environment, and kernel foundations.
- `WarpGasData`/`WarpEnvironmentData` fixed-shape schemas and environment input
  normalization.
- CPU constant and Buck implementations as numerical references.

## Downstream

- E4-F2 and E4-F3 both require E4-F1 and may proceed in parallel only after it.
- E4-F4 requires E4-F1, E4-F2, and E4-F3.
- E4-F5 follows stable integration/latent semantics; E4-F6 requires all physics
  tracks; E4-F7 requires accepted readiness evidence.

## Phase Ordering

`P1 -> P2 -> P3 -> P4 -> P5`. Configuration and validation precede formulas;
formulas precede orchestration; integration precedes contract hardening and
documentation. Each production phase includes its own tests. Production
four-substep scheduling remains E4-F3, but E4-F1's refresh primitive must be
callable at each substep boundary.

## External Dependencies

No new package is planned. Use existing NumPy CPU references, Warp runtime, and
pytest infrastructure.
