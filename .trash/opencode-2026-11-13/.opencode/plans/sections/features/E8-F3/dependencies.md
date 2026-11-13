# Dependencies

## Upstream

- **Parent E8:** defines fixed shapes/order/maps, explicit setup versus replay,
  and prohibits allocation, synchronization, fallback, and automatic recapture.
- **E8-F1 — Captured Resident Timestep:** supplies the lifecycle, compatibility
  signature, invalidation, and exact-resource requirements used at READY and
  capture transitions.
- **E8-F2 — Capture-Safe Setup Boundary:** supplies the authoritative complete
  process/control/temporary requirement inventory and enqueue consumers.
- **Shipped E7 resident execution:** `gpu_session`, `gpu_resources`,
  `resident_communication`, `diagnostics`, `checkpoint`, and RNG lifecycle
  define ownership, schemas, continuation, and fault behavior.
- **Shipped direct Warp kernels and native records:** define exact scratch,
  output, status, communication, and persistent RNG array schemas.

## Downstream and Sibling Handoffs

- **E8-F4:** consumes the exact complete capture set for CPU/uncaptured/captured
  full-loop validation and allocation-forbidden tests.
- **E8-F5:** uses the stable inventory while measuring multi-box scaling.
- **E8-F6:** consumes deterministic role/family logical bytes as one component
  of the broader state, inactive-slot, checkpoint, and future-tape budget.
- **E8-F7:** profiles the final allocation-free replay path and must not replace
  registry ownership or accounting semantics.
- **E8-F8:** documents ownership, setup, byte-report meaning, limitations,
  recapture triggers, the runnable example, and final closeout.

## Phase Ordering

P1 freezes inventory and byte arithmetic before new storage is added. P2 adds
process/control storage using those schemas. P3 completes communication and
diagnostics. P4 can atomically prepare and pin the full set only after all
families exist. P5 integrates that stable set with E8-F1/E8-F2 and updates
documentation. Every phase includes its own tests; there is no standalone test
phase.
