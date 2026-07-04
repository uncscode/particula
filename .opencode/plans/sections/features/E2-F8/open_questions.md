# E2-F8 Open Questions

1. Should CPU coagulation immediately reject `ParticleData.n_boxes != 1`, or
   retain documented transitional box-0 behavior for one release?
   - Preferred answer: reject unsupported multi-box calls to avoid silent misuse.

2. Which public condensation methods provide the best representative coverage
   for multi-box rejection tests?
   - Candidate: `IsothermalCondensationStrategy.step()` plus one pressure-delta
     path that already uses `_require_single_box`.

3. Should docs include caller-managed per-box loop pseudocode?
   - Include only if it can be accurate without creating new helper APIs.

4. Do E2-F2 environment containers need mention in this CPU dynamics boundary
   doc?
   - Likely only as a short cross-reference, unless implementation touches
     environment-aware dynamics paths.

## Questions That Are Out of Scope

- How should full multi-box CPU condensation or coagulation be implemented?
- Should GPU strategy kernels own first-class all-box execution semantics?
- Should the container schema change to represent environment boxes differently?
