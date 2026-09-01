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

## P2 Implementation Record

Issue #1553 implements the first process-specific prepared seams beside their
owning executors. `state_updates.py` binds exact P1 environment or gas requests
and their pinned primary destinations before issuing ordered copy-only writes.
`thermodynamic_updates.py` binds common P1 metadata once while the existing
coordinator remains the owner of mutable cursor and stale markers; each
consumer preparation reads that current state before its vapor or saturation
enqueue. `diagnostics.py` binds the validated closed registration sequence and
pre-resolved outputs before dispatch.

All three prepared paths reject identity or pinning drift during setup and keep
their legacy standalone compatibility paths intact. Their enqueue helpers do
not perform lookup, binding validation, allocation, host readback, or
synchronization. Valid empty schemas remain write-free, and the established
state, thermodynamic, and diagnostics writer ordering is unchanged.

## P3 Implementation Record

Issue #1554 adds frozen, identity-semantic
`PreparedResidentCommunicationBinding` and
`setup_prepared_resident_communication()` beside the legacy communication
executor. Setup proves exact P1 READY ownership, request/duration/node/schedule
retention, primary and published resource-view identities, one closed GAS or
PARTICLES map mode, mode-specific ledgers/statuses/snapshots, and an optional
compatible final-volume array. It rejects drift and unsupported/open maps before
native launch.

`_enqueue_prepared_resident_communication()` selects the already-bound mode
without lookup and invokes only explicit-input native helpers: communication
first, then volume when a sidecar is present. The helpers preserve GAS gated
aggregate-overdraw behavior and PARTICLES planning/commit behavior. An equal
final-volume sidecar returns without modifying primaries, work ledgers, or volume
status lanes; a changed sidecar retains the resident status, concentration
scaling, and volume-update sequence. Compatibility adapters retain legacy
executor and public direct-kernel contracts.

## P4 Implementation Record

Issue #1555 adds the private frozen `_PreparedCondensationCall` in
`particula.gpu.kernels.condensation`. Setup retains the already validated Warp
containers, thermodynamic and optional inputs, fixed duration, scratch/output
sidecars, and required launch inputs. The public `condensation_step_gpu()` keeps
its existing validation order, fallback allocations, supplied-object identity,
and return tuple, then delegates to the prepared enqueue helper.

The helper clears the retained total-transfer output once and runs the unchanged
four equal substeps: vapor-pressure/property refresh, proposal, P2 inventory
finalization, particle mutation, and gas coupling. It has no validation,
allocation, host refresh/readback, synchronization, or registry/resource lookup.
The concrete adapter's private `_PreparedWarpCondensationBinding` holds this
kernel record for its exact resident binding and delegates directly to enqueue.
No public exports, scheduler dispatch, checkpoint/resource schema, or physics
semantics were modified.

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
