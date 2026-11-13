# Infrastructure Reuse

- `RunnableABC.execute()` in `particula/runnable.py:83-106` defines the CPU
  in-place process contract; wrap it in the CPU adapter rather than alter each
  concrete process.
- `RunnableSequence.execute()` in `particula/runnable.py:177-218` is the
  deterministic CPU sequencing reference. E7-F1 describes one process
  execution only and leaves generalized scheduling to E7-F5.
- `WarpParticleData`, `WarpGasData`, and `WarpEnvironmentData` in
  `particula/gpu/warp_types.py:24-184` establish future GPU state types and
  fixed-shape multi-box conventions. Reference these as capabilities without
  importing Warp from the backend-neutral module.
- Explicit upload/restore helpers in `particula/gpu/conversion.py:120-317` and
  `particula/gpu/conversion.py:422-625` establish the no-hidden-transfer rule.
- Existing particle-only `gpu_context` in
  `particula/gpu/conversion.py:629-666` is prior art, not the new complete
  execution context; do not extend its context-manager lifetime into a hidden
  transfer boundary.
- Deferred Warp availability handling and deliberate exports in
  `particula/gpu/__init__.py:36-118` provide patterns for optional dependencies.
- Narrow lazy kernel exports and regression assertions in
  `particula/gpu/kernels/__init__.py` and
  `particula/gpu/tests/kernel_exports_test.py:17-88` provide the export pattern.
- `docs/Examples/gpu_complete_process_sequence.py:380-494` and
  `particula/gpu/tests/process_sequence_test.py:808-872` provide resident-state
  and no-intermediate-restore fixtures for downstream adapters.
- Follow repository typed dataclass/enum, Google docstring, `*_test.py`, and
  explicit `ValueError` validation conventions; keep optional Warp imports lazy.
