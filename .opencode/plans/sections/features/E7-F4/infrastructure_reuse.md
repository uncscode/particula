# Infrastructure Reuse

- `WarpParticleData`, `WarpGasData`, and `WarpEnvironmentData` in
  `particula/gpu/warp_types.py:24-184` already define authoritative fixed-shape,
  multi-box resident schemas. Compose these types; do not duplicate them.
- `to_warp_particle_data()`, `to_warp_environment_data()`, and
  `to_warp_gas_data()` in `particula/gpu/conversion.py:120-377` provide the
  explicit setup boundary and validate device and gas vapor-pressure inputs.
- `from_warp_particle_data()`, `from_warp_gas_data()`, and
  `from_warp_environment_data()` in `particula/gpu/conversion.py:422-626`
  provide synchronized CPU restore behavior. A checkpoint should synchronize
  once, then invoke all three with `sync=False`.
- The particle-only `gpu_context()` in
  `particula/gpu/conversion.py:629-666` is prior art for scoped residency, but
  is intentionally insufficient for complete particle/gas/environment state.
- `CondensationScratchBuffers` and its metadata validator in
  `particula/gpu/kernels/condensation.py:130-209` establish stable-shape,
  same-device sidecar identity and preflight rules.
- `NucleationScratchBuffers`, `NucleationFinalizedDemandBuffers`,
  `NucleationDiagnosticBuffers`, and `NucleationExhaustionBuffers` in
  `particula/gpu/kernels/nucleation.py:246-379` define concrete reusable
  planning and diagnostic resources.
- `docs/Examples/gpu_complete_process_sequence.py:128-179` demonstrates lazy
  loading while retaining concrete records in their owning modules; preserve
  that deliberate export boundary.
- `_allocate_sidecars()` in
  `docs/Examples/gpu_complete_process_sequence.py:182-320` is the allocation
  seed for `(B, N, S)`, `(B, N)`, `(B, S)`, and `(B,)` buffers, including
  coagulation and wall-loss `wp.uint32` per-box RNG arrays.
- `run_example()` in
  `docs/Examples/gpu_complete_process_sequence.py:360-538` demonstrates one
  upload, retained object identities, one synchronization, and one restore.
  Convert the ownership pattern into reusable infrastructure, not a scheduler.
- The conversion guard in
  `particula/gpu/tests/process_sequence_test.py:808-847` detects forbidden
  intermediate restores and should be reused or generalized.
- Five-process composition coverage in
  `particula/gpu/tests/process_sequence_test.py:1651-1679` supplies canonical
  one-box/multi-box fixtures and optional CUDA behavior.
- `particula/gpu/__init__.py:68-118` and
  `particula/gpu/kernels/__init__.py:24-64` show lazy availability checks and
  narrow exports. Public session names should follow E7-F1/E7-F6 policy rather
  than expose every concrete scratch record.
- E7-F1's planned `particula.execution` request/context/state contracts are the
  integration boundary. E7-F6 supplies capability errors, availability,
  fallback, and export policy before this feature ships.
