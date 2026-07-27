# Scope

E7-F2 adds a typed condensation adapter to the execution-context boundary and
proves at least one isothermal or latent-heat selected workflow against the CPU
reference. The feature maps configuration deliberately rather than pretending
that heterogeneous CPU and Warp algorithms are identical.

## In Scope

- Declare supported CPU and Warp condensation capabilities using E7-F1 types.
- Define immutable adapter configuration and execution state for particle, gas,
  environment, thermodynamic/activity inputs, scratch, mass-transfer,
  latent-heat, energy-transfer, and thermal-work resources.
- Delegate CPU execution to `MassCondensation` and GPU execution to the shipped
  `condensation_step_gpu` without rewriting either implementation's physics.
- Support direct-Warp isothermal and latent-heat paths, including fixed four
  substeps, gas coupling, vapor-pressure refresh, identity, and partial-failure
  semantics.
- Reject staggered GPU condensation, unavailable backend/device requests,
  unsupported BAT/activity mappings, malformed state, and mixed environment
  sources before adapter-driven mutation.
- Add CPU/Warp CPU parity, particle-plus-gas conservation, mutation/identity,
  validation-order, no-fallback, and no-transfer tests; add optional CUDA rows.
- Publish narrow user-facing selection documentation and preserve direct APIs.

## Out of Scope

- GPU staggered condensation or new BAT physics.
- Rewriting condensation kernels, CPU strategies, or their numerical schemes.
- Claiming exact CPU/Warp equality where fixed-step algorithms differ; broad
  accuracy, stiffness, or performance claims.
- Resident-session setup/checkpoint ownership (E7-F4), full process scheduling
  and environment ordering (E7-F5), or general fallback policy (E7-F6).
- Hidden conversion, implicit synchronization, silent CPU fallback, dynamic
  resizing/compaction, multi-GPU execution, graph capture, or autodiff.
