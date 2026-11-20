# Direct condensation method decision and downstream gates

The [scientific comparison](../stiffness.md) describes the workload, assumptions and results supporting these decisions.

### Historical Phase Boundary Decision

- This decision preceded the shipped inventory-limited gas-coupled production path.
  The candidate coverage remains deterministic evaluation evidence, not a
  current runtime description.
- The asymptotic candidate remains evidence-only because the tolerance required
  to track the current CPU/explicit reference is materially looser than the
  fixed-count candidate.
- Because the candidate evidence was credible in test-local helpers, no private
  production helper was added to `particula/gpu/kernels/condensation.py`.

## Final Recommendation

### Shipped fixed-four implementation

`fixed_count_substeps_4` is the shipped production integration behavior of
`condensation_step_gpu(...)`, not a future prototype. Each valid call uses four
equal substeps, updates particle mass and gas concentration using the same
inventory-limited transfer, and keeps production calculations device resident.

Why this is the recommended path:

- The historical baseline and candidate study comparison recorded `rtol <= 5e-2` at the baseline
  timestep and maximum relative error `<= 5e-2` only for the named
  `nanometer`, `accumulation_mode`, and `droplet_like` grids.
- Successful calls preserve supplied total-buffer identity, clear that total
  once after preflight, and accumulate inventory-limited transfer. A
  separate work buffer retains the final raw proposal.
- `CondensationScratchBuffers` remains a concrete-module-only sidecar. Its
  supplied active-device, stable-shape `wp.float64` fields may be omitted
  independently, in which case the step allocates fallback buffers.
- This shipped behavior does not establish graph capture/replay or autodiff
  readiness.

### Downstream gates and dependency boundaries

The GPU latent-heat hook supplies a low-level, per-substep latent-heat rate correction with
CPU-oracle/Warp parity. Issue #1272 also ships its caller-owned, signed
`energy_transfer` diagnostic: after successful preflight, its active-device
`wp.float64` `(n_boxes, n_species)` output is overwritten with inventory-limited
transfer times latent heat. It does not add temperature-state
feedback, condensation parity
independent-device plus graph/autodiff evidence, or condensation support documentation strategy/runnable
and final support work.

- **CPU environment-shape dependency:** the recommendation assumes the shipped
  contract for scalar inputs and explicit direct Warp `(n_boxes,)` environment
  arrays. Broader conclusions should not be inferred for different environment
  ownership or shape models.
- **mass-precision study precision dependency:** the recommendation is supported only inside
  the current `float64` / `wp.float64` evidence envelope. It does not approve a
  lower-precision or mixed-precision production migration.
