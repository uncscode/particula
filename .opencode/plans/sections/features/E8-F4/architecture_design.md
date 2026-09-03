# Architecture Design

## High-Level Design

### Delivered P1 qualification boundary

Issue #1567 delivered `qualify_prepared_resident_graph_capture()` in the
concrete-only `particula.execution.graph_capture` module. It accepts the exact
attached E8-F1 READY binding, E8-F2 prepared simulation, E8-F3 published capture
set, and an injected runtime adapter. After fixed identity, lifecycle, registry,
guard, session, device, signature, and resource validation, it performs lazy
runtime → device → capture-API probes and retains one exact
`GraphCaptureNativeCallables` record in a fresh immutable
`PreparedGraphCaptureQualification` result.

`Device.native` is opaque: only CPU and Warp CPU reject before adapter access.
The adapter is the authority for qualification of a non-CPU Warp device. P1
does not invoke any retained callable and has no native graph/exec-handle or
cleanup ownership. It does not open a guard token, capture, enqueue, dispatch,
allocate, transfer, synchronize, or change READY lifecycle state. P2 and P3
remain responsible for native capture, handle publication/release, and replay.

### Planned capture and replay lifecycle

E8-F4 adds a concrete graph owner around the prepared enqueue path; it does not
create a second scheduler. Capture and replay share the same immutable prepared
plan and exact E8-F3 arrays. Host validation is outside capture, and replay does
only bounded metadata checks, token bookkeeping, and one graph launch.

```text
ACTIVE session + pinned registry + closed guard
E8-F1 READY record/signature
E8-F2 PreparedResidentTimestep
E8-F3 CaptureResourceSet
                 |
                 v
prepare_capture() -- exact identity/capability validation (host, read-only)
                 |
                 v
wp.capture_begin(cuda, force_module_load=True)
  enqueue_prepared_timestep()  # fixed twelve-node device sequence only
wp.capture_end() -> opaque graph
                 |
                 v
CapturedResidentPlan(REPLAYABLE, exact binding + graph)
                 |
        replay(duration)
  compare current signature/identities/lifecycle
  begin one resident token
  wp.capture_launch(graph)
  complete token
                 |
       +---------+----------+
       |                    |
 structural drift       launch may fail
       v                    v
  INVALIDATED          graph + session FAULTED
       |                    |
 explicit retire and fresh compatible preparation only
       v
  new READY record -> explicit recapture
```

P1 supplies an internal runtime adapter and frozen native callable vocabulary so
hardware-independent tests can assert qualification order. Future production
capture will use the retained callable vocabulary for `capture_begin`,
`capture_end`, and `capture_launch`; handle publication and cleanup are not part
of the delivered qualification record.

Replay compares metadata before token entry and launch. Mutable payloads and RNG
words are intentionally not compared: they advance in the exact pinned arrays.
Any shape, device, request, schedule, process configuration, communication map,
diagnostic binding, or resource identity drift deterministically invalidates
the record and rejects launch. A graph-launch failure is writer-capable and
faults both graph and resident session; rollback and retry are not promised.

## Data / API / Workflow Changes

- **Data model:** P1 adds exact immutable concrete-only native-callable and
  prepared-qualification records. Captured-plan, opaque graph-handle, and
  teardown records remain future work. Reuse E8-F1 lifecycle/signature and
  invalidation vocabulary rather than creating parallel vocabulary.
- **API surface:** Add direct-module-only setup, replay, invalidate, and teardown
  operations under `particula.execution.graph_capture`. Do not export them from
  `particula.execution`, `particula.gpu.kernels`, or top-level `particula`.
- **Workflow hooks:** E8-F4 consumes E8-F1 lifecycle/signature, E8-F2 prepared
  enqueue, and E8-F3 resource publication. E8-F5 validates the accepted replay
  path; E8-F6 benchmarks its resource/graph lifetime; E8-F7 profiles the
  correctness-qualified path; and E8-F8 documents and closes the epic.
- **Lifecycle:** Capture requires exact ACTIVE/READY/closed bindings. Replay
  requires REPLAYABLE. Finalize, close, fault, explicit teardown, or structural
  drift prevent launch. Recapture always creates a fresh record and native
  handle; old handles are never checkpointed or resurrected.

## Security & Compliance

No network, credential, persistence, or permission boundary changes. The key
robustness control is fail-closed native-handle use: opaque handles never cross
devices, checkpoints, sessions, or restarts; stale or terminal records cannot
launch. Validation exposes identities and metadata only, never pointers,
payloads, or RNG words. Exceptions cannot trigger fallback, migration, retry,
or automatic recapture, and post-launch failures clearly retain the no-rollback
contract.
