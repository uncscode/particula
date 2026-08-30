# Dependencies

## Upstream

- **Parent E8:** supplies the fixed-process-order, stable-shape, explicit
  setup/replay/teardown guardrails and defines E8-F1 as the first ordered track.
- **E7 resident execution:** `gpu_session`, `gpu_resources`,
  `resident_scheduler`, `resident_communication`, `diagnostics`, and
  `checkpoint` provide the exact session, resource, process-order, fault, and
  continuation contracts this feature binds.
- **E6-F5/E6-F6 fixed-capacity policies:** active populations may change inside
  fixed slots without changing captured shapes; exhaustion behavior avoids
  treating every population change as structural invalidation.
- **Persistent RNG lifecycle (E7-F8):** stream initialization/reset is explicit,
  replay never reseeds, and schema-v3 checkpoint continuation remains separate
  from graph-handle lifecycle.
- **Warp capture APIs:** production execution will depend on callable
  `capture_begin`, `capture_end`, and `capture_launch` on a qualified CUDA
  device, but E8-F1 unit contracts must remain testable without CUDA.

## Downstream / Sibling Features

- **E8-F2 — Capture-Safe Setup Boundary:** consumes capability and lifecycle
  gates to move allocation, validation, resource acquisition, and host
  scheduling outside capture.
- **E8-F3 — Reusable Buffer Registry:** must ensure every signature resource is
  preallocated and identity-stable before capture.
- **E8-F4 — Captured Full-Loop Validation:** uses lifecycle and recapture reasons
  to compare CPU, uncaptured GPU, and captured GPU execution.
- **E8-F5/E8-F6:** benchmark and memory models rely on the finalized capture
  lifetime and resource inventory.
- **E8-F7:** publishes user-facing lifecycle, limitations, and recapture
  triggers only after they are executable and validated.
- **E8-F8:** profiles and closes the final implementation; it must not redefine
  E8-F1 lifecycle or invalidation semantics.

## External Dependencies

- Existing `warp-lang` project dependency; no new runtime package is planned.
- `pytest`, `pytest-cov`, Ruff, mypy, and MkDocs from the development extras.

## Phase Ordering

P1 defines capability and the compatibility signature. P2 builds lifecycle and
invalidation only after that signature is stable. P3 integrates recapture gates
with exact resident bindings and therefore depends on P1 and P2. P4 documents
the implemented contract last. Unit tests remain co-located in P1 and P2;
integration tests ship with P3 rather than as a standalone testing phase.
