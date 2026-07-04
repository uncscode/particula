# Open Questions

1. Should `ParticleData.density` remain `(n_species,)` and shared across boxes
   for the whole E2 epic, or should the decision record reserve a future per-box
   material-property representation?
2. Is `ParticleData.volume` authoritative particle-container state, shared
   simulation state, or a field that future `EnvironmentData` may own/mutate?
3. Should `vapor_pressure` be treated as gas state, environment-derived state,
   process scratch/cache, or GPU-only helper state with explicit loss semantics?
4. How should species names be preserved across GPU workflows: CPU-only metadata,
   an index-map sidecar, or a required `names` argument on return conversion?
5. Which humidity or saturation fields are required in the first
   `EnvironmentData` implementation, and should any be `(n_boxes, n_species)`
   instead of `(n_boxes,)`?

## Resolution Policy

- Questions 1-4 should be resolved by E2-F1 because they affect multiple sibling
  tracks.
- Question 5 may be partially resolved by E2-F1 as a shape/ownership rule and
  finalized by E2-F2 when implementing `EnvironmentData`.
- Any unresolved item must name a downstream owner before E2-F1 ships.
