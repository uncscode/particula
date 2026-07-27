# Infrastructure Reuse

- `initialize_coagulation_rng_states()` and the initialization kernel in
  `particula/gpu/kernels/coagulation.py:762,1930` provide the existing
  same-device `(n_boxes,)` `wp.uint32` state boundary. Extend or wrap this path;
  do not create a second coagulation RNG algorithm.
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
- E7-F4's planned `ResidentSession`, `SidecarRegistry`, and versioned checkpoint
  model in `particula/execution/session.py` and
  `particula/execution/checkpoint.py` own allocation, lifecycle, and restore.
- E7-F5's planned `SimulationScheduler` and canonical process graph own whether
  a stochastic process/box executes. RNG policy must consume resolved execution,
  not registration order.
- `_allocate_sidecars()` in
  `docs/Examples/gpu_complete_process_sequence.py:182-320` demonstrates current
  fixed-shape coagulation and wall-loss RNG arrays.
- Conversion and restore helpers in `particula/gpu/conversion.py:120-377,422-626`
  remain the only bulk transfer boundaries; checkpoint should synchronize once
  and restore with `sync=False` thereafter.
- `particula/gpu/tests/kernel_exports_test.py` protects deliberate exports;
  stream metadata may be public through `particula.execution`, while concrete
  kernel helpers remain narrowly exported.
