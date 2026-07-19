# Scope

Delivered a standalone, deterministic CPU/Warp direct-condensation walkthrough
and its regression coverage. The CPU side is an independent NumPy
fixed-four-substep oracle for the bounded low-level contract, not the high-level
CPU strategy or `Runnable` path.

## In Scope

- Independently construct CPU oracle arrays and Warp input arrays from explicit
  fixture constants; do not derive expected values from mutated Warp buffers.
- Exercise a compact fp64 multi-box, multi-species case that makes uptake,
  evaporation, gas coupling, and latent-heat bookkeeping observable.
- Run on Warp `device="cpu"` whenever Warp is installed; allow optional CUDA as
  additive evidence with a clean unavailable-device skip.
- Report independent physics, conservation, and energy acceptance results on
  available Warp CPU and optional CUDA routes. Physics compares final particle
  masses, gas concentrations, P2 total transfer, and exact constant-mode vapor
  pressure; conservation and energy use their own observed-state calculations.
- Exercise no-Warp and force-disabled routes without concrete runtime imports,
  conversion, allocation, synchronization, or kernel execution.
- Document explicit CPU/Warp transfers, caller-owned scratch/energy sidecars,
  synchronization, and recovery after a potentially partial failed call.
- Publish the P3 record with one 14-row deferred-capability table, approved
  downstream owners and entry gates, exact reproduction commands, and a
  regression test for the direct-kernel-only evidence boundary.

## Out of Scope

- Changes to `condensation_step_gpu`, its schemas, kernels, or public API.
- Claims of high-level CPU strategy or `Runnable` equivalence.
- Temperature evolution, adaptive stepping, new activity/surface-tension
  physics, graph replay, broad autodiff, or backend-selection implementation.
- Performance targets, mixed/lower precision migration, or required CUDA CI.
- Broad canonical documentation/index integration, which remains P4 work.
