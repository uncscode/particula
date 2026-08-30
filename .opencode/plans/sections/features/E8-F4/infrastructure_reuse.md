# Infrastructure Reuse

- `ResidentSimulationRequest` and `ResidentSimulationScheduler._validate()` in
  `particula/execution/resident_scheduler.py:99-288` already define the exact
  resident binding, canonical twelve-node schedule, duration agreement, and
  read-only preflight rules. Reuse or factor these checks for preparation; do
  not create a second process-order authority.
- `ResidentSimulationScheduler.execute()` in
  `particula/execution/resident_scheduler.py:465-568` is the authoritative
  uncaptured operation order and token/failure model. E8-F2's prepared enqueue
  must remain numerically shared with this path.
- `ResidentStepGuard`, `_handle_failed_resident_operation`, and
  `_ResidentOperationOutcome` in `particula/execution/gpu_session.py` provide
  exact token ownership and read-only versus writer-may-have-launched faulting.
- `ResidentLifecycle` in `particula/execution/gpu_session.py:278-305` supplies
  ACTIVE, FAULTED, FINALIZED, and CLOSED resident state vocabulary. Graph state
  must react to it rather than redefining resident ownership.
- `GPUResourceRegistry` in `particula/execution/gpu_resources.py` remains the
  authority for same-device, fixed-shape, nonaliasing process, communication,
  diagnostic, and RNG resources. Consume E8-F3's published capture set by exact
  identity and never allocate during capture or replay.
- `ResolvedProcessGraph` and `ResolvedTimestepSchedule` in
  `particula/execution/process_graph.py` and `scheduler.py` supply the canonical
  graph and ordered IDs retained by the prepared plan and compatibility
  signature.
- `ResidentCommunicationExecutor`, state/thermodynamic update coordinators,
  process adapters, and diagnostics executor under `particula/execution/` are
  the existing full-loop seams that E8-F2 converts into private prepared
  enqueue operations.
- `particula/gpu/kernels/tests/condensation_graph_capture_test.py:186-253`
  demonstrates Warp API detection, exact known capability skips, one-time
  cleanup, `ExceptionGroup` preservation, and `capture_launch`; move production
  policy into the concrete graph owner rather than importing test helpers.
- `particula/gpu/kernels/tests/condensation_graph_capture_test.py:359-424`
  provides a reset/capture/replay comparison pattern, identity assertions, and
  conservation checks that can be expanded to the resident full loop.
- E8-F1 defines capability, signature, invalidation, lifecycle, and recapture
  contracts; E8-F2 supplies the immutable prepared sequence; E8-F3 supplies the
  complete pinned resource set and logical-byte report. E8-F4 must consume, not
  duplicate, these sibling contracts.
