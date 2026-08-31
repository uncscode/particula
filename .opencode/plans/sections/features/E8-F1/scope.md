# Scope

E8-F1 delivers the host-side contract layer for deciding whether the exact
resident loop can be captured, tracking a capture record through its lifecycle,
invalidating it when its structural binding changes, and requiring explicit
recapture before further replay. It establishes boundaries for the parent E8
work; it does not yet promise a complete graph-captured physics loop.

## In Scope

- A concrete-only graph-capture capability vocabulary that distinguishes CUDA
  capture support from Warp CPU parity and from runtime/device availability.
- An immutable compatibility signature covering the exact resident session,
  device, dimensions, primary arrays, acquired sidecars, resolved graph and
  schedule, communication configuration/map, diagnostics, and persistent RNG
  resource identities.
- A lifecycle with explicit pre-capture, captured/replayable, invalidated,
  faulted, and closed/retired outcomes, with legal transitions documented and
  unit tested.
- Deterministic invalidation reasons and a compatibility comparison that does
  not launch, synchronize, transfer, allocate, or inspect payload values.
- Explicit recapture eligibility: closed guard, active resident session,
  compatible pinned resources, supported CUDA device, and no hidden fallback.
- A direct-module-only exact resident binding, one-time frozen-request
  attachment, explicit lifecycle-owned gate, and narrow pre-token scheduler
  composition seam.
- Co-located tests in `particula/execution/tests/`; issue #1549 explicitly
  excludes user-documentation changes.

## Out of Scope

- Capturing or replaying the complete twelve-node timestep; executable capture
  work proceeds in E8-F2 and E8-F3 and is validated in E8-F4.
- Moving host validation, allocation, resource acquisition, or scheduler
  planning into a captured graph.
- Automatic recapture, transparent graph replacement, CPU fallback,
  cross-device replay, migration, retries, or rollback after launch.
- Dynamic process selection, shape changes, buffer replacement, communication
  map changes, resizing, or compaction during a capture lifetime.
- Performance targets, benchmarks, memory-budget accounting, examples, and
  final profiling, which belong to E8-F5 through E8-F8.
- New physics, new public top-level exports, broad autodiff, or checkpointing a
  native graph handle.
- User-facing documentation or examples for the binding/gate contract.
