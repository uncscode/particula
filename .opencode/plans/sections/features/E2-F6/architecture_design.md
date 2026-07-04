# E2-F6 Architecture Design

## Design Stance

This feature is a numerical study, not a schema migration. The architecture
keeps the current data containers unchanged while creating reproducible
reference cases and analysis artifacts. Any experimental representation must be
isolated in study helpers or notebooks/scripts and must not become a default
constructor, conversion path, or Warp struct dtype.

## Baseline Model

- CPU baseline: `ParticleData.masses` stores absolute per-species mass in kg as
  `np.float64` with shape `(n_boxes, n_particles, n_species)`.
- GPU baseline: `WarpParticleData.masses` stores absolute per-species mass in kg
  as `wp.float64` with the same logical shape.
- Derived quantities such as radius, total mass, effective density, and mass
  fractions are computed from absolute mass and density.

## Study Harness

The study should separate three layers:

1. **Case generation:** deterministic particle/gas states spanning the dynamic
   range. These may reuse small-particle fixtures and benchmark scaling
   conventions.
2. **Candidate evaluation:** adapters that project the baseline cases into
   candidate precision/representation forms and back into comparable physical
   quantities.
3. **Metric reporting:** conservation, relative/absolute small-mass error,
   radius error, clamping frequency, memory footprint, and runtime/throughput.

## Conservation Reference

Prefer CPU conservation-limited functions in
`particula/dynamics/condensation/mass_transfer.py` as the reference because the
current GPU condensation path applies transfer and clamps negative particle
masses without clearly coupling gas depletion in the reviewed path.

## Output Contract

The report must explicitly state whether to keep absolute `fp64`, adopt a mixed
path, or investigate a representation alternative before schema changes. The
recommendation must include the evidence threshold and the cases that justify
it.
