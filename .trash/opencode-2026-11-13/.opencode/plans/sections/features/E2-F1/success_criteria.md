# Success Criteria

## Done Signal

- A schema decision record exists for particle, gas, and environment state.
- Shape documentation covers all current containers:
  - `ParticleData`
  - `GasData`
  - `WarpParticleData`
  - `WarpGasData`
- Shape documentation also covers future environment containers sufficiently for
  E2-F2 and E2-F3 to implement them.
- Downstream tracks E2-F2 through E2-F9 have explicit field ownership and handoff
  decisions.

## Reviewable Outcomes

- Every current container field has an owner, shape, dtype, mutability note, and
  CPU/GPU transfer note.
- Single-box workflows are documented as retaining the leading `n_boxes`
  dimension.
- Multi-box, particle-resolved, and binned workflows have unambiguous shape
  rules.
- Ambiguous fields such as `vapor_pressure`, species `name`, `partitioning`,
  `density`, and `volume` have decisions or clearly assigned follow-up owners.

## Validation Outcomes

- Documentation links resolve.
- Existing relevant container tests are referenced or run successfully by the
  implementation issue.
- No downstream feature must infer ownership from source code alone.
