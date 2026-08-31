# Implementation Tasks

## Backend

- [x] Create `particula/execution/graph_capture.py` as a concrete-only module;
  keep all names absent from package and top-level exports.
- [x] Define exact immutable capability and compatibility carriers and
  deterministic field validation. Lifecycle carriers remain P2 work.
- [x] Implement a metadata-only capture-capability resolver that separates
  Warp runtime/device availability, CUDA capture API support, and structural
  resident compatibility; do not import Warp from dependency-neutral modules.
- [x] Build a compatibility signature from one exact
  `ResidentSimulationRequest`, including session/device/dimensions, graph and
  schedule, primary arrays, resource views, diagnostics, communication, and RNG
  identities.
- [x] Compare current bindings against a signature in deterministic field order
  and return/raise the first documented recapture reason without payload reads.
- [x] Implement host-only immutable lifecycle transitions for ready, captured,
  invalidated, faulted, retired, and closed states with explicit close metadata.
- [x] Preserve the first incompatible P1 drift reason and retained capability/
  signature identities through lifecycle successors; do not perform native work.
- [x] Require active session plus closed guard/registry gates before capture or
  recapture eligibility; reject finalized, faulted, and closed sessions.
- [x] Preserve existing scheduler failure classification: read-only rejection
  leaves state active; possible writer launch faults the exact binding without
  rollback, fallback, or retry.
- [x] Add only the narrow integration seam needed in
  `particula/execution/resident_scheduler.py`; do not fork its twelve-node order
  or move adapter dispatch into this feature.

## Tooling / Tests

- [x] Add `particula/execution/tests/graph_capture_test.py` using `*_test.py`
  naming, hardware-free capability/import fakes, and Warp-guarded real-request
  signature cases.
- [x] Parametrize all signature drift triggers: backend/native device,
  dimensions, container/array identity, graph, schedule/order, process config,
  sidecar, diagnostics, communication map/buffer, and RNG resource.
- [x] Assert payload-only changes and active/free slot changes do not invalidate
  stable-shape compatibility.
- [x] Assert CPU capture is unsupported rather than emulated, CUDA absence
  cleanly skips hardware rows, and unexpected allocation/runtime failures are
  not converted to capability skips.
- [x] Extend `particula/execution/tests/full_loop_test.py` with exact resident
  binding and recapture-gate integration assertions, without claiming captured
  physics parity before E8-F4.
- [x] Add/retain graph-capture test assertions proving internals are
  concrete-only.
- [x] Add hardware-free P2 coverage for legal and illegal transitions, exact
  argument validation, first-reason retention, failure classification, and a
  lifecycle subprocess that forbids Warp and resident-module imports.
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
