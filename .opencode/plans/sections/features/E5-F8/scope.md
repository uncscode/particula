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
- Compare final particle masses, gas concentrations, total finalized transfer,
  final raw proposal, and energy against the independent oracle on available
  Warp CPU and optional CUDA routes.
- Exercise no-Warp and force-disabled routes without concrete runtime imports,
  conversion, allocation, synchronization, or kernel execution.
- Document explicit CPU/Warp transfers, caller-owned scratch/energy sidecars,
  synchronization, and recovery after a potentially partial failed call.

## Out of Scope

- Changes to `condensation_step_gpu`, its schemas, kernels, or public API.
- Claims of high-level CPU strategy or `Runnable` equivalence.
- Temperature evolution, adaptive stepping, new activity/surface-tension
  physics, graph replay, broad autodiff, or backend-selection implementation.
- Performance targets, mixed/lower precision migration, or required CUDA CI.
- Deferred-capability ownership tables, broad documentation/index changes, and
  category-isolation reporting infrastructure planned in later E5-F8 phases.
