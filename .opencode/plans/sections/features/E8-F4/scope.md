# Scope

E8-F4 implements the runtime owner that consumes the exact E8-F2 prepared
resident timestep and E8-F3 capture resource set. E8-F4-P1 now provides the
read-only qualification boundary; later phases record and replay the fixed
device sequence while E8-F1 compatibility and lifecycle contracts remain valid.

## In Scope

- A concrete-only captured-plan/controller record retaining the exact prepared
  plan, compatibility signature, capture resource set, native device, opaque
  graph handle, and lifecycle state.
- Explicit `capture_begin` / prepared enqueue / `capture_end` setup on qualified
  CUDA devices, with cleanup that consumes an active capture exactly once.
- Replay preflight for exact session, registry, guard, prepared plan, signature,
  schedule, resource-set, graph-handle, device, and lifecycle identity.
- One graph launch per accepted timestep under resident token and writer-failure
  semantics; no validation, allocation, readback, or host scheduling inside the
  captured sequence.
- Deterministic invalidation, fault, close, teardown, and explicit recapture
  eligibility behavior for structural and resident lifecycle changes.
- Multi-timestep three-way validation across CPU reference, uncaptured Warp,
  and captured CUDA for identical supported process configurations.
- Co-located unit, lifecycle, integration, CUDA pass-or-clean-skip, export, and
  documentation-contract tests.

## Delivered in E8-F4-P1 (issue #1567)

- `particula/execution/graph_capture.py` provides the direct-import-only
  prepared qualification controller, immutable native-callable and
  qualification records, lazy adapter capability probes, and exact identity
  retention for the READY binding, prepared simulation, and capture set.
- Qualification rejects CPU and Warp-CPU before adapter access; the adapter
  performs ordered runtime, device, and capture-API checks for opaque non-CPU
  Warp native devices.
- The delivered boundary neither invokes native callables nor captures,
  dispatches, opens a guard token, allocates, transfers, synchronizes, publishes
  a handle, or performs cleanup. READY remains unchanged on every outcome.
- Tests were added in `particula/execution/tests/graph_capture_test.py` and
  `particula/execution/tests/exports_test.py`; the new names remain intentionally
  absent from package and top-level exports.

## Delivered in E8-F4-P3 (issue #1569)

- `particula/execution/graph_capture.py` now provides the direct-import-only
  `replay_captured_resident_graph()` boundary. It accepts only an authentic
  P2-issued captured record, checks opaque-handle provenance by identity, and
  validates the exact captured binding, capture publication, device, lifecycle,
  and duration before token entry.
- Each accepted call opens one resident token, invokes the retained native
  `capture_launch` once with the exact opaque handle, and completes that token.
  It neither dispatches prepared host operations nor allocates, probes,
  transfers, reads back, synchronizes, reseeds, falls back, retries, or
  recaptures.
- Launch and completion failures use existing writer-capable resident cleanup
  and graph/session fault classification; rollback is not promised. Preflight
  and `begin_step()` failures are read-only and do not launch.
- Coverage was added in `particula/execution/tests/graph_capture_test.py` for
  authentic-handle provenance, exact drift/lifecycle/duration rejection,
  one-launch/one-token success, and writer-failure handling. Public exports are
  unchanged.

## Delivered in E8-F4-P4 (issue #1570)

- `particula/execution/graph_capture.py` is the sole owner of issued captured
  handles, teardown, native release, and graph lifecycle transition. Its common
  teardown path removes every matching issued record before one release for
  structural drift, writer fault, finalization, close/discard, and retirement.
- Resident lifecycle owners use a lazy private notification with the exact
  session, registry, and closed guard; they neither import graph-capture types
  at module load nor inspect opaque handles or change graph state themselves.
- `gpu_session.py`, `checkpoint.py`, and `gpu_resources.py` notify the graph
  owner for authoritative writer faults and terminal transitions. Published
  stream initialization now requires and validates the exact closed guard.
- Lifecycle regressions cover stale provenance before token entry or launch,
  exact-once release including release failures, read-only preservation, fault
  and terminal ordering, attachment/context rejection, and fresh-record-only
  recapture. Checkpoint payloads and public exports remain unchanged.

## Out of Scope

- Dynamic shapes, process-order selection, communication-map replacement, or
  resource replacement during replay.
- Automatic recapture, CPU fallback, retry, migration, hidden transfer,
  synchronization, checkpoint serialization of graph handles, or rollback.
- Graph capture on Warp CPU; Warp CPU remains the uncaptured contract baseline.
- New process physics, resizing, compaction, distributed execution, autodiff,
  performance targets, memory-budget publication, examples, or profiling; those
  belong to E8-F5 through E8-F8.
- Public exports from `particula.execution`, `particula.gpu.kernels`, or the
  top-level package.
