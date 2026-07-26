# Dependency Map

## Inbound

- Shipped Epic E5 supplies direct GPU coagulation, persistent RNG, charge
  handling, inactive-slot behavior, and GPU validation conventions.
- Existing direct GPU condensation supplies fixed-shape gas coupling,
  inventory finalization, and reusable sidecar patterns.
- Existing CPU wall-loss strategies are the neutral and charged physics oracle.
- Warp is required for implementation and Warp CPU validation; CUDA is optional.

## Outbound

- Epic G remains pending until E6-F1 through E6-F8 are Shipped/completed with
  dated phases and E6-F9 P1-P4 command evidence passes. E6-F2, E6-F5, E6-F6,
  and E6-F8 currently block it; Epic G owns backend selection, scheduling,
  resident-loop APIs, and transport.
- Future performance, graph-capture, and differentiability work consumes these
  low-level contracts but is not part of E6.

## Sequencing

1. E6-F1 precedes E6-F2.
2. E6-F3 precedes E6-F4.
3. E6-F5 precedes E6-F6.
4. E6-F7 requires E6-F5 and E6-F6.
5. E6-F8 requires E6-F5, E6-F6, and E6-F7.
6. E6-F9 requires E6-F1 through E6-F8.

E6-F1, E6-F3, and E6-F5 may begin in parallel. CPU references always precede
their GPU parity features, and every feature ships its own unit tests.
