# Infrastructure Reuse

- `ResidentSimulationRequest` and `ResidentSimulationScheduler._validate()` in
  `particula/execution/resident_scheduler.py:99-289` already define the exact
  session/registry/guard, graph, schedule, duration, diagnostics, and process
  resource checks. Extract or call this logic during setup rather than
  duplicating weaker validation.
- `ResidentSimulationScheduler.execute()` in
  `particula/execution/resident_scheduler.py:465-568` is the authoritative
  twelve-node order and writer-failure boundary. Prepared enqueue must preserve
  this order rather than introducing a second scheduler.
- E8-F1's concrete-only `particula.execution.graph_capture` lifecycle and
  compatibility signature are the authority for READY/CAPTURED/REPLAYABLE,
  exact identities, invalidation reasons, and fault handling.
- `GPUResourceRegistry.validate_*` and published views in
  `particula/execution/gpu_resources.py` provide identity, schema, device, and
  nonaliasing validation. E8-F3 will extend the reusable inventory; E8-F2 must
  consume views without acquiring or replacing them during enqueue.
- `ResidentCommunicationExecutor.validate()` in
  `particula/execution/resident_communication.py:108-143` is a metadata-only
  setup seam. Its resident primitives at lines 145-230 already avoid P1
  configuration scans and should be split into prepared enqueue calls.
- `ResidentDiagnosticsExecutor.validate()` in
  `particula/execution/diagnostics.py:326-343` already separates validation from
  launch; reuse it when constructing a prepared diagnostic plan.
- `ResidentStateUpdateExecutor` in
  `particula/execution/state_updates.py:344-447` identifies the current capture
  blockers: transient `wp.zeros` and `.numpy()` validation at lines 317-341.
  Preserve these checks in setup and retain only copy launches in enqueue.
- `ResidentThermodynamicUpdateCoordinator` and the private saturation kernel in
  `particula/execution/thermodynamic_updates.py:211-511` provide canonical
  refresh ordering. Setup should resolve the fixed refresh sequence so enqueue
  has no Python freshness decisions or repeated binding validation.
- Resident adapters in `particula/execution/process_adapters.py:242-412` and
  `particula/execution/adapters/{condensation,coagulation}.py` preserve exact
  resource identities. Extend them with prepared execution states rather than
  exposing new package APIs.
- Public direct kernels under `particula/gpu/kernels/` remain the validation
  wrappers. Refactor their established launch sequences into private prepared
  enqueue functions used by both the wrapper and resident path.
- `particula/gpu/kernels/tests/condensation_graph_capture_test.py` supplies the
  existing CUDA capability/skip pattern and proves that host validation
  readback is the current capture blocker.
