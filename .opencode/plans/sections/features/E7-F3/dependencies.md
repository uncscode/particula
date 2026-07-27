# Dependencies

## Upstream

- **E7-F1 — Backend-selection and execution-context API:** supplies typed
  requests, capabilities, state/adapter protocols, registry, and result/mutation
  vocabulary. E7-F3 must not invent a parallel selector.
- **E7-F6 — CPU fallback, capability errors, exports, and API-stability policy:**
  freezes backend availability, explicit fallback prohibition/boundaries,
  unsupported errors, lazy Warp loading, and public export policy.
- Shipped CPU `Coagulation`/`BrownianCoagulationStrategy` behavior is the CPU
  reference; shipped `coagulation_step_gpu` and its tests are the Warp contract.
- Fixed-shape Warp particle/environment containers and explicit conversion
  helpers from E2 remain unchanged.

## Downstream

- **E7-F5 — Deterministic full-process scheduling:** depends on the selected
  Brownian process node, result semantics, and stable failure boundary.
- **E7-F8 — Persistent per-box RNG streams and restart semantics:** extends the
  T3 per-box RNG seam with stream identity, disabled/reordered box behavior, and
  checkpoint/restart rules.
- **E7-F9 — Diagnostics and closeout:** consumes E7-F3 parity, conservation,
  no-transfer, optional CUDA, API, and documentation evidence.
- E7-F4 may host T3 outputs/RNG in resident session state, but E7-F3 does not
  depend on E7-F4 to define the standalone adapter contract.

## Phase Ordering

`P1 -> P2 -> P3 -> P4 -> P5 -> P6`.

P1 waits for E7-F1 and E7-F6 contracts. P2 freezes resource ownership before P3
dispatch. P4 completes unsupported/failure semantics before P5 integration

## External Dependencies

- Warp is required for Warp CPU and optional CUDA execution; CPU-only import
  and operation must remain possible when Warp is unavailable.
- NumPy/pytest provide independent deterministic and statistical references.
- CUDA hardware is optional evidence and never a mandatory CI dependency.
