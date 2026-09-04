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

### Delivered P2 capture lifecycle

Issue #1568 adds the concrete-only capture owner around the prepared enqueue
path; it does not create a second scheduler. `CapturedResidentGraph` retains
the exact qualification/binding identities and the opaque native end handle.
`capture_prepared_resident_graph()` validates the exact READY qualification,
calls native begin, uses the private capture-safe dispatcher for the retained
twelve operations, calls native end, revalidates, and only then transitions and
publishes CAPTURED state. Replay remains deferred to P3.

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
CapturedResidentGraph(CAPTURED, exact binding + opaque graph)
                  |
         P3 delivered: replay(duration)
   authenticate P2-issued handle and compare exact retained binding
  begin one resident token
   wp.capture_launch(graph) exactly once
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
   teardown unregisters before one native release
   new READY record -> explicit recapture
```

P1 accepts a caller-injected runtime adapter and retains its frozen native
callable vocabulary so hardware-independent tests can assert qualification
order. P2 invokes only `capture_begin()`, `capture_end()`, and private
`capture_release(handle)` on post-end rejection or dispatch cleanup; it never
inspects the handle or invokes instantiate/launch. Its dispatcher performs only
the frozen canonical operation calls and deliberately omits validation, guard
tokens, thermodynamic bookkeeping, and cleanup. P3 supplies replay without
adding scheduler dispatch or native-handle lifecycle work.

`replay_captured_resident_graph()` first requires exact P2 issuance and an
identity-matching opaque handle, then revalidates the retained binding and
duration before token entry. Mutable payloads and RNG words are intentionally
not compared: they advance in the exact pinned arrays. Any shape, device,
request, schedule, process configuration, communication map, diagnostic
binding, or resource identity drift rejects before launch. Accepted replay does
exactly one `begin_step`, one native `capture_launch`, and one completion. A
launch or completion failure is writer-capable and faults graph/session state;
rollback and retry are not promised.

### Delivered P4 teardown and terminal notification

Issue #1570 centralizes teardown in `graph_capture.py`. The graph owner retains
the private exact resident binding, removes issued records before attempting
native release, and transitions to a nondispatchable lifecycle successor before
surfacing a release error. This makes stale provenance fail before guard-token
entry or `capture_launch`; repeated notification or terminal calls cannot invoke
a second release. Structural drift preserves the canonical first invalidation
reason, writer failure classifies the graph fault, finalization/close close it,
and retirement preserves the invalidated-to-retired renewal path.

`gpu_session.py`, `checkpoint.py`, and `gpu_resources.py` reach that owner only
through lazy private notification with the exact session/registry/closed-guard
triple. Finalization tears down the issued record before copying or caching its
ordinary handle-free checkpoint. Close/discard and writer failure also notify
before their resident transition. Stream initialization requires the same exact
closed guard, so stream-writer failures retain the authoritative
fault-and-teardown path. No checkpoint schema, restart rule, export, automatic
recapture, retry, rollback, fallback, or opaque-handle transfer was added.

## Data / API / Workflow Changes

- **Data model:** P1 adds exact immutable concrete-only native-callable and
  prepared-qualification records; P2 adds immutable `CapturedResidentGraph`
  retaining the exact CAPTURED successor and opaque end handle. Reuse E8-F1
  lifecycle/signature and invalidation vocabulary rather than creating parallel
  vocabulary.
- **API surface:** P2 adds direct-module-only
  `capture_prepared_resident_graph()` under
  `particula.execution.graph_capture`; P3 adds direct-module-only
  `replay_captured_resident_graph()` there. P4 adds private notification and
  teardown seams only; do not export concrete names from `particula.execution`,
  `particula.gpu.kernels`, or top-level `particula`.
- **Workflow hooks:** E8-F4 consumes E8-F1 lifecycle/signature, E8-F2 prepared
  enqueue, and E8-F3 resource publication. E8-F5 validates the accepted replay
  path; E8-F6 benchmarks its resource/graph lifetime; E8-F7 profiles the
  correctness-qualified path; and E8-F8 documents and closes the epic.
- **Lifecycle:** Capture requires exact ACTIVE/READY/closed bindings. Replay
  requires an authentic CAPTURED record, exact ACTIVE session, and closed guard.
  Finalize, close, fault, explicit teardown, or structural
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
