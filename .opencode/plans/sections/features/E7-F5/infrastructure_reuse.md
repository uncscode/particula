# Infrastructure Reuse

- `RunnableSequence` in `particula/runnable.py:132-218` supplies deterministic
  CPU composition prior art. Preserve CPU adapter semantics; do not pass Warp
  containers through the `Aerosol`-typed runnable contract.
- E7-F1's planned `particula.execution` request, capability, context, and result
  types are the neutral scheduler boundary.
- P1 (#1492) reuses only `Process`, `CapabilityRequirements`,
  `CONDENSATION_PROCESS`, and `CONDENSATION_CAPABILITY_MATRIX` from
  `particula.execution`; it deliberately does not reuse/import E7-F4 resource
  views or any Warp/GPU module.
- E7-F2's condensation adapter delegates to `MassCondensation.execute()` and
  `condensation_step_gpu()`; reuse its typed state/resource view.
- E7-F3's Brownian adapter delegates to `Coagulation.execute()` and
  `coagulation_step_gpu()` while retaining persistent RNG/output identities.
- E7-F4 plans `ResidentSession` in `particula/execution/gpu_session.py` and the
  resource registry in `particula/execution/gpu_resources.py`; use its
  `begin_step()`, process views, faulting, and `complete_step()` hooks.
- `dilution_step_gpu`, `wall_loss_step_gpu`, and `nucleation_step_gpu` are the
  deliberate exports in `particula/gpu/kernels/__init__.py:24-64`; adapters must
  call these boundaries rather than private kernels.
- `WarpParticleData`, `WarpGasData`, and `WarpEnvironmentData` in
  `particula/gpu/warp_types.py:24-184` define fixed resident schemas.
- `refresh_vapor_pressure_gpu()` in
  `particula/gpu/kernels/thermodynamics.py:318-377` is the on-device refresh
  primitive after temperature changes.
- Environment normalization in `particula/gpu/kernels/environment.py:304-419`
  establishes exact shape/device and positive-finite validation patterns.
- `docs/Examples/gpu_complete_process_sequence.py:128-538` and
  `_allocate_sidecars()` provide five-process call and resource-allocation
  seeds; the example is explicitly not the production scheduler.
- Transfer guards and five-process fixtures in
  `particula/gpu/tests/process_sequence_test.py:808-872,1651-1677` plus call
  order/failure tests in
  `particula/gpu/tests/gpu_complete_process_sequence_example_test.py:494-801`
  should be generalized instead of duplicated.
