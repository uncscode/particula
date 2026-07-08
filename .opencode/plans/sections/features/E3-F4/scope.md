# Scope

## In Scope

- Decide the public import path for direct low-level kernel step functions:
  - Preferred baseline: `from particula.gpu.kernels import ...`.
  - Optional narrow flattening: re-export only `condensation_step_gpu` and
    `coagulation_step_gpu` from `particula.gpu` if maintainers choose top-level
    discoverability.
- Add import/export regression tests for the selected path and `__all__`
  behavior where applicable.
- Create a runnable quick-start example showing both condensation and
  coagulation low-level calls.
- Use existing CPU data containers and explicit helpers:
  - `ParticleData`
  - `GasData`
  - `to_warp_particle_data` / `from_warp_particle_data`
  - `to_warp_gas_data` / `from_warp_gas_data`
  - `to_warp_environment_data` where environment examples are needed
  - `gpu_context`
  - `WARP_AVAILABLE`
- Add docs/tests proving the example degrades cleanly when Warp or CUDA is not
  available.
- Add troubleshooting guidance for Warp/CUDA/device mismatch and mixed
  environment inputs.

## Out of Scope

- High-level backend selection, automatic CPU/GPU dispatch, or runtime backend
  abstraction.
- Hidden CPU-to-GPU or GPU-to-CPU transfers inside kernels or runnables.
- Changing kernel numerical behavior except as required by already-dependent
  feature `E3-F1` RNG persistence.
- Publishing raw Warp kernel internals such as `*_kernel` launch functions as a
  broad stable API unless explicitly approved during import-path review.
- Performance optimization beyond quick-start validation and smoke coverage.

## Parent and Sibling Context

Parent epic: `E3`.

Sibling tracks already drafted include:

- `E3-F1`: seed-once persisted coagulation RNG states; prerequisite for any
  repeated coagulation quick-start behavior.
- `E3-F2`: mixed-scale coagulation sampling evidence/hardening.
- `E3-F3`: coagulation benchmark and one-thread-per-box scaling evidence.

This feature depends on `E3-F1` before documenting repeated coagulation calls
with persisted `rng_states`.
