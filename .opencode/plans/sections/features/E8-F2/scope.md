# Scope

E8-F2 delivers the T2 setup/enqueue split for the complete resident timestep.
One-time setup performs host-side validation, normalization, dependency
resolution, and prepared-plan construction. The prepared path enqueues only
the already-resolved fixed device operations. Both boundaries are
concrete-only and retain exact caller-owned resident identities.

## In Scope

- **Completed P1 (#1552):** provide the direct-import-only READY preparation
  boundary and immutable aggregate metadata carrier in
  `particula/execution/resident_enqueue.py`; extract read-only diagnostics and
  communication validators; and share complete-loop metadata validation with
  the scheduler without executor construction or lifecycle mutation.
- **Completed P2 (#1553):** add P1-bound prepared setup and enqueue-only seams
  for resident environment/gas state replacement, thermodynamic vapor-pressure
  and saturation refresh, and diagnostics. Setup validates exact ownership and
  identity chains; enqueue consumes pre-bound arrays only while preserving
  writer order, thermodynamic cursor/freshness semantics, and empty no-ops.
   Existing standalone executor/coordinator paths remain compatible.
- **Completed P3 (#1554):** bind one exact P1 READY closed-map GAS or PARTICLES
  communication resource family, duration, and optional final-volume sidecar in
  `particula/execution/resident_communication.py`; dispatch fixed native
   communication then volume barriers through
   `particula/gpu/kernels/communication.py`. Equal final volumes are write-free;
   changed final volumes preserve established resident evolution semantics.
- **Completed P4 (#1555):** split the direct condensation call into private
  `_PreparedCondensationCall` setup and device-only enqueue in
  `particula/gpu/kernels/condensation.py`, then retain it through the concrete
  `_PreparedWarpCondensationBinding` in
  `particula/execution/adapters/condensation.py`. Public validation, fallback
  allocation, output identity, and four-substep gas-coupled P2 physics remain
  unchanged; no public, scheduler, checkpoint, or resource-schema change was
   introduced.
- **Completed P5 (#1556):** split direct coagulation, dilution, and wall-loss
  calls into private prepared/enqueue seams in their owning kernel modules, and
  retain them through concrete resident coagulation/process adapter bindings.
  Existing public validation, return/identity, no-op, selected-lane, and
  persistent-RNG contracts remain unchanged; resident coagulation remains
  Brownian-only. No public, scheduler, checkpoint, resource-schema, or physics
  change was introduced.
- Define immutable prepared-enqueue carriers tied to the exact E8-F1 capture
  lifecycle signature and `ResidentSimulationRequest`.
- Move repeated scheduler metadata checks and executor construction into an
  explicit setup operation that is forbidden during capture.
- Provide validation-free, allocation-free enqueue seams for resident state
  updates, thermodynamic refreshes, diagnostics, communication, volume
  evolution, condensation, coagulation, dilution, wall loss, and nucleation.
- Preserve the canonical twelve-node order, refresh windows, selected-box
  semantics, persistent RNG advancement, empty/no-op behavior, and resident
  writer-failure classification.
- Retain existing public direct-kernel entry points as validate-then-enqueue
  wrappers; prepared resident paths remain private/concrete-only.
- Add CPU/Warp-CPU contract tests for setup and enqueue structure, plus
  CUDA-gated capture smoke evidence that skips cleanly when unavailable.

## Out of Scope

- P1 does not capture, replay, enqueue, dispatch, create scheduler/executor
  instances, enter a guard token, acquire resources, inspect payloads, transfer,
  synchronize, select a device, or mutate lifecycle state.
- Owning the graph lifecycle, compatibility signature, invalidation, or
  recapture policy established by E8-F1.
- Completing the reusable registry inventory or changing buffer ownership;
  that is E8-F3 scope.
- Three-way scientific parity, scaling benchmarks, memory modeling, examples,
  profiling, or performance claims owned by E8-F4 through E8-F8.
- Dynamic shapes, process selection, communication-map changes, allocation,
  payload readback, synchronization, RNG reseeding, fallback, retry, automatic
  recapture, device migration, resizing, compaction, new physics, or public API
  expansion inside the enqueue boundary.
