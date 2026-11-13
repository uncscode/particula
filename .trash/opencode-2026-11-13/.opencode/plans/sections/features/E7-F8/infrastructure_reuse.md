# Infrastructure Reuse

- Existing coagulation and wall-loss kernels establish the caller-owned,
  same-device `(n_boxes,)` `wp.uint32` state-array conventions. P1 does not
  wrap, replace, or change their RNG algorithms; it only writes validated
  caller-owned initial state through `wp.copy`.
- `coagulation_step_gpu()` persistent-state behavior in
  `particula/gpu/kernels/coagulation.py:2215-2222,2341-2356` already seeds only
  on explicit `initialize_rng=True`; preserve its return and mutation contract.
- Wall-loss RNG validation, initialization, and persistence in
  `particula/gpu/kernels/wall_loss.py:444-468,507-510,894-1015` provide the
  second process boundary and its shape/dtype/device checks.
- Existing persistent resource usage in
  `particula/gpu/tests/process_sequence_test.py:1409-1493,1789-1808` supplies
  multi-call fixtures for coagulation and wall loss.
- E7-F3's typed Brownian adapter contract requires one setup seed followed by
  `initialize_rng=False`; use its process-resource view rather than bypassing
  backend selection.
- Resident session/resource and checkpoint seams retain their existing ownership,
  lifecycle, allocation, and restore responsibilities. P1 deliberately neither
  acquires arrays from nor binds a `StreamRegistry` to those seams.
- E7-F5's planned `SimulationScheduler` and canonical process graph own whether
  a stochastic process/box executes. RNG policy must consume resolved execution,
  not registration order.
- `_allocate_sidecars()` in
  `docs/Examples/gpu_complete_process_sequence.py:182-320` demonstrates current
  fixed-shape coagulation and wall-loss RNG arrays.
- Conversion and restore helpers in `particula/gpu/conversion.py:120-377,422-626`
  remain the only bulk transfer boundaries; checkpoint should synchronize once
  and restore with `sync=False` thereafter.
- `particula/execution/tests/exports_test.py` and
  `particula/tests/execution_exports_test.py` protect the direct-only boundary:
  `particula.execution.rng` and its concrete names remain absent from public
  package exports.
