# Dependencies

## Upstream

- **Parent E8:** defines the strict setup/replay/teardown boundary and prohibits
  allocation, synchronization, fallback, or dynamic structure during replay.
- **E8-F1 — Captured Resident Timestep:** supplies capture capability,
  lifecycle states, compatibility signatures, invalidation, recapture, and
  writer-fault contracts consumed by preparation.
- **Shipped E7 resident execution:** `gpu_session`, `gpu_resources`,
  `resident_scheduler`, `resident_communication`, `diagnostics`, state updates,
  thermodynamic updates, checkpoint continuation, and persistent RNG streams.
- **Shipped direct Warp kernels:** authoritative physics and launch ordering for
  condensation, coagulation, dilution, wall loss, nucleation/exhaustion,
  communication, volume evolution, and thermodynamic refresh.

## Downstream and Sibling Handoffs

- **E8-F3 — Reusable Buffer Registry:** consumes the prepared-process inventory
  and must provide every temporary, validation, selected-lane, diagnostic, and
  RNG sidecar with stable identity before preparation completes.
- **E8-F4 — Captured Full-Loop Validation:** compares CPU, uncaptured prepared,
  and captured prepared sequences; it must not invoke validation during capture.
- **E8-F5/E8-F6:** benchmark and memory accounting use the final prepared
  sequence and exact resource inventory.
- **E8-F7:** profiles the validated setup/replay path without redefining the
  setup or enqueue contract.
- **E8-F8:** documents setup, capture, replay, teardown, recapture triggers,
  the runnable example, and final closeout.

## Phase Ordering

P1 defines the common prepared-record invariants. P2 and P3 adapt non-physics
resident operations. P4 through P6 refactor process launches in dependency
order while preserving their public wrappers. P8 shipped the resulting
development documentation for issue #1559 after focused assertions (22 passed
in 0.09s) and `mkdocs build --strict` (exit 0 in 14.67s). P7 remains pending and
may compose the full sequence only after all nodes have prepared enqueue seams.
Tests remain co-located with every implementation phase; there is no standalone
testing phase.
