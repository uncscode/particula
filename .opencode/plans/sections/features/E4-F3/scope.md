# Scope

E4-F3 promotes reusable scratch ownership first, then the selected fixed-count
condensation integrator after E4-F1 establishes on-device thermodynamic refresh.

## In Scope

- **Delivered in P1 / issue #1292:** Add concrete-module-only
  `CondensationScratchBuffers` in `particula/gpu/kernels/condensation.py` with
  optional fixed-shape fp64 work-transfer, total-transfer, dynamic-viscosity,
  and mean-free-path fields.
- **Delivered in P1 / issue #1292:** Validate the complete supplied sidecar and
  reject `mass_transfer` overlap with supplied scratch transfer fields before
  allocation, normalization, refresh, launch, clear, or particle mutation.
- **Delivered in P1 / issue #1292:** Support partial sidecars; allocate only
  omitted fields through the compatibility path. Complete sidecar reuse needs no
  allocation of the four stable shapes.
- **Delivered in P2 / issue #1293:** Execute exactly four unconditional
  substeps of `time_step / 4`, refreshing E4-F1 and environment state every
  substep from current particle mass.
- **Delivered in P2 / issue #1293:** Mass-clamp each proposal before applying
  it, accumulate applied transfer in the resolved total buffer, and retain the
  final raw proposal in work storage.
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
