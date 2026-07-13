# Outcomes and Guardrails

- **Primary Outcome:** Deliver a conservative, thermodynamically refreshed GPU
  condensation kernel whose supported modes agree with independent CPU
  references on Warp CPU and, when available, CUDA.
- **Secondary Goals:**
  - Preserve fixed array shapes and caller-owned reusable scratch buffers.
  - Provide signed whole-call mass and latent-energy diagnostics.
  - Publish explicit model, device, ownership, and tolerance contracts.
  - Keep every feature track independently tested as it lands.
- **Guardrails / Non-Goals:**
  - No hidden CPU/GPU synchronization or implicit container transfers.
  - No adaptive stepping, staggered/Gauss-Seidel integration, or data-dependent launch counts.
  - No BAT activity, general strategy-object execution in Warp, or automatic backend selection.
  - No high-level `Aerosol`/`Runnable` GPU integration, coagulation work, or fp64 schema change.
  - T6 consolidates evidence; it does not defer unit tests from T1-T5.
