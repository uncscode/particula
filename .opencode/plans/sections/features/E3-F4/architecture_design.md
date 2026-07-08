# Architecture Design

## Public API Shape

The low-level direct-kernel API should remain explicit and module-scoped. The
recommended stable import path is:

```python
from particula.gpu.kernels import condensation_step_gpu, coagulation_step_gpu
```

Do not add broad top-level `particula.gpu` exports for this quick-start. A later
separate API decision may choose narrow re-exports, but the plan direction here
is to use `particula.gpu.kernels` for direct step-function imports and keep raw
Warp launch functions such as `apply_*_kernel` or `*_mass_transfer_kernel` out
of broad user-facing exports. Do not expose high-level backend selection or
imply automatic transfers.

## Transfer Boundary Design

The quick-start must make boundaries visible:

1. Construct CPU containers (`ParticleData`, `GasData`).
2. Check `WARP_AVAILABLE` before importing or executing Warp-dependent paths
   that require Warp runtime behavior.
3. Transfer to Warp data explicitly via helper functions or `gpu_context` for
   particle data.
4. Call `condensation_step_gpu` and `coagulation_step_gpu` only with
   GPU-resident data and compatible direct environment inputs.
5. Transfer back explicitly with `from_warp_*` helpers for inspection.

## Device and Environment Handling

- Default examples should use `device="cpu"` for deterministic accessibility.
- CUDA snippets should be optional and skip/branch when CUDA is unavailable.
- All Warp arrays passed to a kernel must live on the same device.
- Examples must not pass scalar `temperature`/`pressure` at the same time as
  `environment=`.

## Dependency Design

The coagulation example should account for `E3-F1` RNG semantics. If this
feature lands before `E3-F1`, keep the quick-start to a single coagulation call
or explicitly avoid promising repeated-call persisted RNG behavior.
