# Open Questions

Status: reviewed and answered on 2026-07-08.

## Resolved Decisions

1. The final public quick-start should import direct step functions from
   `particula.gpu.kernels`. That package already exports `condensation_step_gpu`
   and `coagulation_step_gpu`.
2. Raw lower-level symbols such as `apply_coagulation_kernel` should remain
   excluded from top-level `particula.gpu.__all__`. If top-level re-exports are
   added later, limit them to the two step functions and cover them with export
   tests.
3. Put the quick-start beside the existing data-container/GPU foundation
   examples unless implementation reveals a stronger need for a separate direct
   kernels path.
4. After E3-F1, the minimum coagulation snippet should show a caller-owned
   `rng_states` buffer initialized once, passed into repeated
   `coagulation_step_gpu` calls, and retained by the caller between steps.
5. Troubleshooting should live in both the runnable example comments and the
   feature documentation. The example should cover immediate failure modes;
   the docs should explain broader CUDA/Warp availability and transfer-boundary
   issues.
