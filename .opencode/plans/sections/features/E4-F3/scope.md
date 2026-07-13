# Scope

E4-F3 promotes the selected fixed-count condensation integrator and reusable
scratch ownership after E4-F1 establishes on-device thermodynamic refresh.

## In Scope

- Execute exactly four unconditional substeps of `time_step / 4`.
- Recompute transfer from current particle state and refresh applicable E4-F1
  thermodynamic state during every substep.
- Add reusable fixed-shape fp64 work, total-transfer, dynamic-viscosity, and
  mean-free-path buffers where required by the production implementation.
- Validate every caller-owned buffer before clearing buffers, launching kernels,
  or mutating particle state.
- Return total transfer accumulated over all four substeps while preserving
  caller-buffer identity.
- Promote deterministic, nonnegative, finite, stiffness-bound, shape, device,
  mutation-order, and no-allocation regression coverage from issue #1272.
- Preserve scalar, direct Warp-array, hybrid, and `WarpEnvironmentData` inputs.

## Out of Scope

- Adaptive or data-dependent substep counts.
- Gas concentration updates or gas/particle conservation accounting (E4-F5).
- Activity and effective surface-tension models owned by E4-F2.
- Latent-heat correction and energy diagnostics owned by E4-F4.
- Changes to CPU/GPU container schemas, precision, or hidden transfer behavior.
- New diagnostics; none were requested.
