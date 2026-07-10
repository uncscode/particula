# Open Questions

Status: reviewed and answered on 2026-07-08.

## Resolved Decisions

1. The supported public import path is
   `from particula.gpu.kernels import condensation_step_gpu,
   coagulation_step_gpu`.
2. `particula.gpu` remains intentionally non-reexporting for these step
   functions, and `particula/gpu/tests/kernel_exports_test.py` now guards that
   negative contract directly.
3. Raw lower-level symbols such as `apply_coagulation_kernel` stay out of the
   package-level public surface; `particula.gpu.kernels.__all__` is narrowed to
   the two supported step functions only.
4. Put the quick-start beside the existing data-container/GPU foundation
   examples unless implementation reveals a stronger need for a separate direct
   kernels path.
5. After E3-F1, the minimum coagulation snippet should show a caller-owned
   `rng_states` buffer initialized once, passed into repeated
   `coagulation_step_gpu` calls, and retained by the caller between steps.
6. Troubleshooting should live in both the runnable example comments and the
   feature documentation. The example should cover immediate failure modes;
   the docs should explain broader CUDA/Warp availability and transfer-boundary
   issues.
