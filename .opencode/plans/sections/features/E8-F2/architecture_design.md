# Architecture Design

## High-Level Design

E8-F2 introduces preparation and enqueue layers beneath the existing resident
scheduler and direct-kernel wrappers. Preparation is allowed to inspect host
metadata, launch validation scans, read validation status, normalize controls,
and build immutable launch records. Enqueue accepts only an exact prepared
record and emits a predetermined sequence of device operations.

```text
exact ACTIVE session + pinned registry + closed guard
resolved twelve-node request + E8-F1 READY capture record
                         |
                         v
prepare_resident_timestep()                 HOST / ONE TIME
  validate lifecycle, schedule, identities, schemas, values
  resolve node order, refresh windows, modes, selected boxes
  bind all primary arrays, controls, kernels, and sidecar identities
                         |
                         v
PreparedResidentTimestep (immutable; exact identities/signature)
                         |
             +-----------+-----------+
             |                       |
             v                       v
uncaptured enqueue once       Warp capture_begin(CUDA)
                                     |
                                     v
enqueue_prepared_timestep()                 DEVICE ENQUEUE ONLY
  communication -> volume -> updates -> thermal refresh
  -> condensation -> coagulation -> dilution -> wall loss
  -> nucleation -> diagnostics
                                     |
                                     v
                              capture_end / replay (E8-F1)
```

Each public direct-kernel API remains a `validate/normalize -> private enqueue`
wrapper. The prepared resident plan calls the same private enqueue primitive
after setup has proved all preconditions. This avoids divergent physics while
keeping unsafe bypasses concrete-only and inaccessible to ordinary callers.

## P1 Implementation Record

Issue #1552 implements the READY-only aggregate metadata boundary in
`particula.execution.resident_enqueue`. Its frozen, `eq=False`
`PreparedResidentTimestep` retains the exact request, attached binding,
lifecycle/signature, session, registry, closed guard, device, dimensions,
graph, schedule, canonical node-ID tuple, duration, primary-array tuple, and
published-resource tuple without copying payloads. The boundary validates the
exact attached ACTIVE/pinned/closed chain, requires READY, compares the E8-F1
signature before and after setup, and leaves lifecycle ownership with
`graph_capture`.

`resident_scheduler._validate_complete_resident_timestep_metadata()` is now the
shared read-only complete-loop validator. It reuses extracted functional
validators in `diagnostics.py` and `resident_communication.py`, so preparation
does not construct their executors or a scheduler and does not invoke the
CAPTURED-only scheduler graph gate. Capture/replay, token entry, resource
acquisition, payload inspection, transfer, synchronization, lifecycle mutation,
and device selection remain excluded from P1.

## Data / API / Workflow Changes

- **Data model:** Add frozen prepared request records for the full timestep and
  process-specific launch arguments. Records retain exact arrays, sidecars,
  normalized scalar controls, dimensions, selected lanes, and E8-F1
  compatibility identity; they do not copy or own device state.
- **API surface:** Keep all new preparation and enqueue names direct-import-only
  under `particula.execution` or private to kernel modules. Do not modify
  `particula.execution.__all__`, `particula.gpu.kernels.__all__`, or top-level
  exports. Existing direct entry-point signatures and validation order remain
  compatible.
- **Workflow hooks:** E8-F1 authorizes preparation and graph lifecycle. E8-F2
  produces the complete prepared sequence. E8-F3 supplies every stable reusable
  array before capture. E8-F4 validates three-way results; later tracks consume
  the same path for evidence and documentation.
- **Failure model:** Preparation is read-only with respect to primary physics
  state and leaves lifecycle READY on rejection. Enqueue is writer-capable from
  its first launch; failures fault the resident session and capture lifecycle
  according to E8-F1, with no rollback, retry, fallback, or automatic recapture.
- **Dynamic controls:** Only values stored in already-bound device arrays may
  change without preparation. Python scalar or structural changes require a new
  prepared record and follow E8-F1's explicit recapture decision.

## Security & Compliance

No network, credential, serialization, or permission boundary changes. The
main robustness requirement is fail-closed access to validation-free enqueue:
prepared records must be exact, unforgeable by ordinary public APIs, tied to the
current lifecycle signature, and rejected before launch after identity drift.
Opaque graph handles and prepared records are not checkpointed or portable
across devices. Enqueue must not expose device payloads or RNG words, silently
fall back to CPU, or perform hidden host transfer/readback.
