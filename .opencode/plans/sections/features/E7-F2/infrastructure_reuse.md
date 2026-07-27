# Infrastructure Reuse

## Execution Boundary

- Reuse E7-F1's planned `particula.execution` request, capability matrix,
  context, adapter protocol, and result semantics. Do not introduce a second
  backend selector.
- Consume E7-F6's availability, explicit-fallback, capability-error, export, and
  stability policy. Adapter failures propagate; they are never caught to retry
  another backend.
- Reuse `RunnableABC` delegation and `MassCondensation` in
  `particula/dynamics/particle_process.py:458-557` for the CPU path.

## Direct-Warp Boundary

- Wrap, but do not alter, `condensation_step_gpu` in
  `particula/gpu/kernels/condensation.py:1814-2003`.
- Preserve `WarpParticleData`, `WarpGasData`, and `WarpEnvironmentData` schemas
  from `particula/gpu/warp_types.py` and explicit transfer ownership from
  `particula/gpu/conversion.py`.
- Reuse `ThermodynamicsConfig`, `CondensationActivitySurfaceConfig`, and
  `CondensationScratchBuffers` as concrete GPU-side inputs. They remain narrow
  concrete-module types rather than broad top-level exports.
- Preserve the existing four-substep vapor-pressure refresh, inventory-limited
  gas coupling, transfer accumulator, optional latent heat, and energy output.

## Tests and Examples

- Adapt independent NumPy fixtures and parity cases from
  `particula/gpu/kernels/tests/condensation_test.py` and its support module.
- Reuse no-intermediate-transfer and process-composition spies from
  `particula/gpu/tests/process_sequence_test.py` and
  `gpu_complete_process_sequence_example_test.py`.
- Keep `docs/Examples/gpu_complete_process_sequence.py` as low-level prior art,
  not a production selector or scheduler.
