# Architecture Design

## High-Level Design

Add a concrete-only `particula.execution.graph_capture` contract layer around
the existing resident request. The layer records capture compatibility and
lifecycle; it does not replace `ResidentSimulationScheduler`, own resident
arrays, or perform hidden setup. Executable Warp capture/replay is a downstream
consumer of this contract.

```text
resolved ResidentSimulationRequest
  + exact ACTIVE session / pinned registry / closed guard
  + CUDA capture capability decision
        |
        v
build CaptureCompatibilitySignature (metadata + identities only)
        |
        v
CaptureLifecycle: READY -> CAPTURED -> REPLAYABLE
        |                        |
        | structural drift       | writer may have launched and failed
        v                        v
    INVALIDATED               FAULTED
        |
        | explicit validate-for-recapture on a closed active binding
        v
      READY (new record; old graph retired)

Any state -> CLOSED only through explicit teardown.
```

Compatibility is structural. The signature includes backend/native device,
resident dimensions, primary-array identities, graph and schedule identities
and canonical node IDs, process configuration identities, all published
sidecars and diagnostics, communication mode/map/work buffers, and RNG sidecar
identities. Mutable payload values—including active-slot occupancy, gas
concentration, RNG words, and scalar controls stored in already-pinned device
arrays—are not signature fields.

The comparison returns a deterministic invalidation reason instead of silently
recapturing. Shape, device, process order, request object, communication map,
or buffer identity changes invalidate. Session finalization, close, or fault
make the record non-recapturable. A compatible active binding may create a new
capture record only through an explicit setup call after the old record is
retired.

## Data / API / Workflow Changes

- **Data model:** Introduce immutable capture capability, compatibility
  signature, invalidation reason, and lifecycle records. Use closed enums and
  exact dataclass validation consistent with resident execution modules.
- **API surface:** Keep the module concrete-only; do not add names to
  `particula.execution.__all__` or top-level `particula`. Candidate operations
  are metadata-only capability resolution, signature construction/comparison,
  explicit invalidation, recapture eligibility, and teardown.
- **Workflow hooks:** E8-F2 consumes `READY` to perform capture-safe setup;
  E8-F3 guarantees all signature resources are preallocated; E8-F4 validates
  execution parity; E8-F7 profiles the validated replay path; E8-F8 documents
  user-facing recapture triggers. The
  existing uncaptured scheduler remains authoritative for process order and
  writer-failure semantics.
- **Error behavior:** Unsupported capture capability is distinct from invalid
  compatibility and unavailable runtime/device errors. Read-only validation
  failures leave the active session and capture record unchanged. A replay-side
  failure after a writer may have launched faults both the capture record and
  resident session, with no rollback or retry guarantee.

## Implemented P1 Boundary

Issue #1547 delivered the declaration-only portion in
`particula/execution/graph_capture.py`. It provides capability outcomes for
CPU, Warp CPU, unavailable runtime/device, unsupported API, and availability;
the resolver uses caller-provided lazy probes and imports no Warp module.

It also provides immutable identity-only signatures and deterministic first
drift comparison for the request, session, device, dimensions, containers,
primary arrays, resource views, graph, schedule/order, diagnostics,
communication, configurations, and RNG sidecars. The implementation retains
existing request-owned references only; it does not create lifecycle records or
perform capture/replay. Those operations remain P2-P3 scope.

## Security & Compliance

No new network, credential, file-deserialization, or permission boundary is
introduced. Robustness is fail-closed: opaque native graph handles are never
serialized, checkpointed, or trusted across devices; invalid records cannot
launch; and exceptions must not trigger automatic CPU fallback or recapture.
Identity and schema checks must remain bounded metadata operations and must not
expose device payloads or RNG words through diagnostics.
