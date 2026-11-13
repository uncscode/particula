# Dependencies

## Upstream

- **E7-F3 / T3 — Backend-selected Brownian coagulation:** supplies the typed
  adapter and persistent coagulation RNG resource contract. E7-F8 specializes
  stream identity, box independence, reset, and restart behavior.
- **E7-F4 / T4 — GPU-resident session and checkpoints:** supplies lifecycle,
  sidecar registry, fixed dimensions, explicit checkpoint/finalize boundaries,
  and opaque mutable-resource restoration.
- **E7-F5 / T5 — Deterministic full-process scheduler:** supplies canonical
  process order, resolved enabled nodes/boxes, and post-launch fault semantics.
- Inherited foundations include E7-F1 execution context, E7-F6 fail-closed
  capability/fallback policy, shipped direct coagulation/wall-loss kernels, Warp,
  and NumPy reference/test support.

## Downstream

- **E7-F9 / T9** consumes stream diagnostics, full-loop restart regressions,
  optional CUDA evidence, documentation, and Epic G closeout validation.
- E7-F7 transport remains a sibling: box communication may change physical
  state but must not redefine logical RNG identities or consume these streams.

## Phase Ordering

P1 freezes identity and seed derivation before process integration. P2 and P3
bind separate process streams. P4 exposes controlled lifecycle operations after
ownership is stable. P5 proves box-level invariance. P6 then freezes checkpoint
schema and restart continuation over the integrated behavior. P7 documents the
validated contract. No E7-F8 phase should start implementation until the
corresponding E7-F3/E7-F4/E7-F5 seam exists or its interface is jointly frozen.

## External Dependencies

- Warp CPU is required when Warp is installed; CUDA hardware is optional.
- No new third-party RNG, serialization, distributed-runtime, or cryptography
  dependency is planned.
