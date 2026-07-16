# Scope

Deliver a standalone, deterministic CPU/Warp direct-condensation walkthrough,
its regression coverage, and a durable deferred-capability ownership table.
The CPU side is an independent NumPy fixed-four-substep oracle for the bounded
low-level contract, not the high-level CPU strategy or `Runnable` path.

## In Scope

- Independently construct CPU oracle arrays and Warp input arrays from explicit
  fixture constants; do not derive expected values from mutated Warp buffers.
- Exercise a compact fp64 multi-box, multi-species case that makes uptake,
  evaporation, gas coupling, and latent-heat bookkeeping observable.
- Run on Warp `device="cpu"` whenever Warp is installed; allow optional CUDA as
  additive evidence with a clean unavailable-device skip.
- Report physics parity for final particle masses and gas concentrations.
- Report per-box/per-species concentration-weighted particle-plus-gas inventory
  conservation separately from physics parity.
- Report signed `energy_transfer = finalized mass transfer * latent_heat`
  separately from both parity and conservation.
- Record owner, entry gate, and non-claim for phase-aware surface tension, BAT
  activity, `thermal_work` consumption and temperature feedback, adaptive
  stepping, high-level integration, graph capture/replay, broad autodiff,
  general CPU-strategy parity, and other explicitly deferred boundaries.
- Add regression and documentation-link checks with each phase.

## Out of Scope

- Changes to `condensation_step_gpu`, its schemas, kernels, or public API.
- Claims of high-level CPU strategy or `Runnable` equivalence.
- Temperature evolution, adaptive stepping, new activity/surface-tension
  physics, graph replay, broad autodiff, or backend-selection implementation.
- Performance targets, mixed/lower precision migration, or required CUDA CI.
- Coagulation work owned by E5-F1 through E5-F7 or roadmap closeout owned by
  E5-F9.
