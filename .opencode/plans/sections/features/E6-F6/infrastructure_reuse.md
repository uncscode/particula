# Infrastructure Reuse

- `ParticleData` in `particula/particles/particle_data.py:47-80` owns fixed
  float64 masses, concentration/weight, charge, density, and per-box volume;
  preserve all array shapes and identities.
- `ParticleData.total_mass` and derived properties at
  `particula/particles/particle_data.py:167-217` provide independent CPU
  accounting inputs; policy code should use explicit weighted reductions.
- `WarpParticleData` in `particula/gpu/warp_types.py:24-78` mirrors the same
  fields and shape contract; extend behavior, not the struct schema.
- E6-F5's planned `particula.particles.slot_management` and
  `particula.gpu.kernels.slot_management` APIs provide strict active/free
  predicates, ascending free indices, exact int32 diagnostics, and atomic
  activation. Consume those APIs rather than reclassifying slots.
- E6-F5 architecture lines 14-35 establish read-only preflight followed by one
  mutation boundary and exact diagnostics; use the same pattern for policy
  planning and commit.
- `ParticleResolvedSpeciatedMass.add_mass` at
  `particula/particles/distribution_strategies/particle_resolved_speciated_mass.py:150-177`
  demonstrates fill-free-first behavior. Reuse the ordering concept, but never
  reuse its `np.concatenate` resize fallback.
- Coagulation's fixed-slot merge/deactivation path in
  `particula/gpu/kernels/coagulation.py:1345-1415` is prior art for conserving
  mass/charge while clearing donor slots.
- Direct GPU validation and caller-owned sidecar patterns in
  `particula/gpu/kernels/coagulation.py` and condensation tests should guide
  shape, dtype, device, identity, and failure-order checks.
- `docs/Theory/Technical/Dynamics/Nucleation_Equations.md:157-181` defines why
  weighted computational particles and an exhaustion policy are required.
- `docs/Features/Roadmap/data-oriented-gpu.md:1219-1232` is authoritative for
  fixed slots, no timestep allocation, exhaustion handling, and diagnostics.
