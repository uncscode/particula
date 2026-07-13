# Architecture Design

For each of exactly four substeps, production code computes a raw per-particle
transfer from current gas and thermodynamic state, then applies a deterministic
pipeline before any state mutation:

1. Zero transfer for nonpartitioning species and inactive particle slots.
2. Clamp negative transfer to each particle's available species mass.
3. Reduce concentration-weighted positive and negative transfer independently
   into `(n_boxes, n_species)` fp64 scratch.
4. Add same-substep evaporation to available gas and calculate a `[0, 1]`
   positive-transfer scale for each box/species.
5. Apply positive scales, derive the exact finalized aggregate transfer, and
   update particle mass and gas concentration with opposite signs.
6. Accumulate finalized transfer for return and E4-F4 energy bookkeeping, then
   refresh gas-dependent physics before the next substep.

Reductions must be deterministic for the supported backend. All required
scratch is fixed-shape and caller-reusable; validation occurs before launches or
mutation. Gas concentration remains kg/m3 and particle mass remains per
particle, so concentration weighting and box-volume conventions must match the
CPU reference explicitly. No atomically independent gas clamps are allowed:
the same finalized transfer is authoritative for both states, diagnostics, and
latent heat.
