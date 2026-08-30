# Infrastructure Reuse

- `Capability`, `CapabilityRequirements`, `CapabilityMatrix`, and
  `ExecutionRequest` in `particula/execution/__init__.py:151-364` provide the
  dependency-neutral capability vocabulary. Reuse these value-object and exact
  declaration patterns rather than probing Warp during metadata construction.
- `resolve_availability()` and `_WarpAvailabilityProvider` in
  `particula/execution/availability.py:121-177,302-402` already separate runtime
  and device availability from structural capability. Graph-capture capability
  checks should preserve this ordering and should not treat Warp CPU as capture
  capable.
- `ResidentSession`, `ResidentDimensions`, and `ResidentLifecycle` in
  `particula/execution/gpu_session.py:181-291,693-757` define immutable shape,
  device, identity, and active/faulted/terminal rules. A capture record should
  bind these objects; it should not own or replace them.
- `ResidentStepGuard.assert_step_closed()` and failure classification in
  `particula/execution/gpu_session.py:951-1139,1273-1313` provide the lifecycle
  gates and post-writer fault semantics to mirror around capture, replay, and
  recapture operations.
- `GPUResourceRegistry._session_signature()` and
  `validate_pinned_session()` in
  `particula/execution/gpu_resources.py:559-605` already detect lifecycle,
  dimension, container, primary-array, and device drift without payload reads.
  Extend this pattern with graph-specific schedule, sidecar, communication, and
  diagnostic identities.
- `GPUResourceRegistry` resource manifests in
  `particula/execution/gpu_resources.py:344-464` enumerate the fixed-shape
  condensation, coagulation, wall-loss, nucleation, and communication roles
  that later capture setup must pin.
- `ResolvedProcessGraph` and `resolve_canonical_topological_order()` in
  `particula/execution/process_graph.py:161-184,411-469` provide immutable graph
  declarations and deterministic ordering. Capture compatibility must retain
  the resolver-produced graph and schedule by identity, not rebuild ordering.
- `ResidentSimulationRequest` and `ResidentSimulationScheduler._validate()` in
  `particula/execution/resident_scheduler.py:99-193,229-289` already enforce the
  exact complete twelve-node resident binding. Reuse this preflight boundary;
  do not duplicate process order in an independent capture scheduler.
- The existing Warp API harness in
  `particula/gpu/kernels/tests/condensation_graph_capture_test.py:186-253`
  records `capture_begin`, `capture_end`, and `capture_launch` availability and
  precise cleanup behavior. Reuse its known capability distinctions while
  moving production lifecycle policy into `particula.execution`.
- Schema-v3 checkpoint and RNG continuation remain owned by
  `particula/execution/checkpoint.py` and `gpu_resources.py`; native graph
  handles must not become checkpoint payloads, and replay must never reseed.
