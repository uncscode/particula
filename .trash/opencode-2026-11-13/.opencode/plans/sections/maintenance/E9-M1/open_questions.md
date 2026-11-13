# Open Questions and Review Decisions

Authority: [E9 review decisions](../../epics/E9/appendix.md#review-decisions-2026-09-27).
The decisions below and final validation have been accepted; Kyle approved
the M1 handoff on 2026-11-10. See `validation_evidence.md`.

- [x] **Q1 species alignment:** D3 chooses explicit process-owned lane mapping,
  full gas-order environment lanes and preservation of nonparticipating material.
  Nonempty unique gas names and expected configuration order are validated;
  unnamed particle chemistry remains caller-declared, not inferred from shape.
- [x] **Q2 concentration convention (later maintainer clarification):** PMF
  bins are number per m³ and radius PDF is dn/dr per m⁴, both at fixed V=1 m³;
  neither divides by V. Particle-resolved stores represented counts and divides
  by positive finite V once. PDF bulk properties integrate by trapezoids over
  an explicit positive increasing radius grid; PMF/resolved bulk properties sum.
  Shared distribution_type metadata uses the existing three-value vocabulary.
  D2 explicitly authorizes necessary CPU/GPU correction and metadata work.
- [x] **Q3 access API:** D4 chooses particles/gas/environment properties with
  validated setters and all-three replace_data; copies detach, raw fields are
  writable, derived properties remain fresh. No redundant get_*/set_* family.
- [x] **Q4 helpers/validation placement:** reuse existing derived properties;
  provide consumer-backed normalization/population helpers. Proposed concrete
  aerosol_validation.py owns shared aggregate structure; builder creation
  validates kind/volume; an explicit read-only ParticleData validation method
  gates post-mutation interpretation. Processes own physics, mapping and
  distribution compatibility. No reconstruction to validate.
- [x] **Q5 governance:** Kyle/Gorkowski approves semantics and final M1 evidence.

## Completed M1 evidence

P1/P2 decisions are approved in the dated appendix update. P3's explicit
representation metadata, builder admission, population helpers and transfer
rejection, and P4's concrete structural checker are delivered on the accumulated
branch. P5 adds the data-only mixed/all-false, non-unit-volume integration
regression and the bounded developer reference. The required checks pass at
`0db7e42b2`; Kyle approved the resulting M1 handoff at PR revision `a7c1eb63c`
on 2026-11-10. No M2 aggregate implementation or downstream physics migration
is delivered.
