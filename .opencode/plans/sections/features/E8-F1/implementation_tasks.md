# Implementation Tasks

## Backend

- [ ] Create `particula/execution/graph_capture.py` as a concrete-only module;
  keep all names absent from package and top-level exports.
- [ ] Define exact immutable capability and compatibility carriers, closed
  lifecycle/invalidation enums, and deterministic field validation.
- [ ] Implement a metadata-only capture-capability resolver that separates
  Warp runtime/device availability, CUDA capture API support, and structural
  resident compatibility; do not import Warp from dependency-neutral modules.
- [ ] Build a compatibility signature from one exact
  `ResidentSimulationRequest`, including session/device/dimensions, graph and
  schedule, primary arrays, resource views, diagnostics, communication, and RNG
  identities.
- [ ] Compare current bindings against a signature in deterministic field order
  and return/raise the first documented recapture reason without payload reads.
- [ ] Implement lifecycle transitions for ready, captured/replayable,
  invalidated, faulted, retired, and closed states with explicit teardown.
- [ ] Require active session plus closed guard/registry gates before capture or
  recapture eligibility; reject finalized, faulted, and closed sessions.
- [ ] Preserve existing scheduler failure classification: read-only rejection
  leaves state active; possible writer launch faults the exact binding without
  rollback, fallback, or retry.
- [ ] Add only the narrow integration seam needed in
  `particula/execution/resident_scheduler.py`; do not fork its twelve-node order
  or move adapter dispatch into this feature.

## Tooling / Tests

- [ ] Add `particula/execution/tests/graph_capture_test.py` using `*_test.py`
  naming and hardware-free fakes for capability and lifecycle unit tests.
- [ ] Parametrize all signature drift triggers: backend/native device,
  dimensions, container/array identity, graph, schedule/order, process config,
  sidecar, diagnostics, communication map/buffer, and RNG resource.
- [ ] Assert payload-only changes and active/free slot changes do not invalidate
  stable-shape compatibility.
- [ ] Assert CPU capture is unsupported rather than emulated, CUDA absence
  cleanly skips hardware rows, and unexpected allocation/runtime failures are
  not converted to capability skips.
- [ ] Extend `particula/execution/tests/full_loop_test.py` with exact resident
  binding and recapture-gate integration assertions, without claiming captured
  physics parity before E8-F4.
- [ ] Add/retain export tests proving graph-capture internals are concrete-only.
- [ ] Run focused assertions with coverage disabled, then run the untargeted
  `.opencode/tools/run_pytest.py` suite for repository-configured full-package
  coverage; focused-target coverage is invalid evidence.
- [ ] Run Ruff formatting/checks and mypy for changed Python modules.

## Documentation

- [ ] Update the Epic H roadmap text to distinguish E8-F1's lifecycle contract
  from executable capture and replay delivered by later tracks.
- [ ] Record the full recapture-trigger table and no-fallback/no-checkpointed-
  graph-handle constraints in `AGENTS.md` for downstream agents.
- [ ] Mark E8-F1 plan phases and changelog accurately after implementation.
