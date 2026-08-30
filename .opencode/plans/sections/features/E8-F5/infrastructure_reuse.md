# Infrastructure Reuse

- `ResidentSimulationScheduler` and `ResidentSimulationRequest` in
  `particula/execution/resident_scheduler.py` remain the authoritative
  uncaptured twelve-node loop and failure semantics.
- The complete fixture and independent inventory pattern in
  `particula/execution/tests/full_loop_test.py:94-107,264-275,894-972` supplies
  canonical order, stable-identity, conservation, and lifecycle assertions.
- Dynamic multi-box fixtures, logical box IDs, device gating, snapshots, and RNG
  checks in `particula/execution/tests/multi_box_loop_test.py:92-95,133-216,
  307-344,636-841` should be generalized rather than duplicated.
- The extensive-amount NumPy oracle and closed GAS-map rows in
  `particula/execution/tests/transport_loop_test.py:130-158,161-268` establish
  communication and volume-evolution expectations.
- `ResidentDiagnosticsPlan` and diagnostic registrations used in
  `particula/execution/tests/full_loop_test.py:380-443` provide the full output
  matrix: snapshots, total mass, particle number, latent energy, and residual.
- `GPUResourceRegistry` in `particula/execution/gpu_resources.py` owns stable,
  nonaliasing process, communication, diagnostic, and RNG sidecars.
- `ResidentStepGuard`, session lifecycle, checkpoint/restart, and RNG operations
  in `gpu_session.py`, `checkpoint.py`, and `rng.py` provide rejection and
  continuation contracts.
- E8-F1 lifecycle/signature, E8-F2 prepared enqueue, E8-F3 capture resource set,
  and E8-F4 graph owner are direct upstream seams; tests must consume them rather
  than mock a parallel capture architecture.
- Follow `particula/gpu/kernels/tests/condensation_graph_capture_test.py` for
  CUDA capability checks and clean skips, but keep full-loop expectations
  independent from production helpers.
