# Open Questions and Review Decisions

Authority: [E9 review decisions](../../epics/E9/appendix.md#review-decisions-2026-09-27).
Resolved planning choices below do not mark M1 phases complete.

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

## Evidence still required in M1

P1 reconciles exact helper contracts, clarified PDF grid/quadrature,
metadata-construction policy and the complete CPU/GPU consumer ledger against
the later D1 clarification and D2–D3. Characterize V=0.25/1/4 for resolved
counts only, and PMF/PDF at fixed V=1 independently; do not bless legacy
agreement as correctness. P2 publishes the D4 replacement acceptance matrix.
P3/P4 implement bounded helpers/metadata admission with adjacent tests and
transfer implications; P5 supplies literal final-revision results and approval.
No normalization implementation or passing characterization is claimed here.
