# Scope and Constraints

## In Scope

- Numeric GPU thermodynamic configuration and per-substep vapor-pressure refresh.
- Constant and Water Buck vapor-pressure modes.
- Ideal and kappa activity plus selected effective surface-tension physics.
- Fixed four-substep production orchestration with reusable scratch.
- Latent-heat correction and signed energy diagnostics.
- Partitioning gates, gas mutation, inventory bounds, and per-box/species conservation.
- Warp CPU parity, optional CUDA parity, graph/buffer checks, bounded autodiff evidence.
- Support matrix, examples, troubleshooting, and focused reproduction commands.

## Out of Scope

- BAT or arbitrary Python strategy objects in Warp kernels.
- Adaptive or staggered integration and hidden transfer/synchronization.
- High-level backend dispatch and unrelated GPU processes.
- Production data-type or container-schema migration.

## Constraints

- Preserve `(n_boxes, n_particles, n_species)` and `(n_boxes, n_species)` layouts.
- Preserve fp64 mass and concentration storage and explicit species ordering.
- Normalize environment inputs through `particula/gpu/kernels/environment.py`.
- Use independent one-box CPU references for multi-box GPU parity.
- Validate supplied buffers, shapes, devices, and physical values before mutation.
