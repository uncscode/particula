# Open Questions and Review Decisions

Authority: [E9 review decisions](../../epics/E9/appendix.md#review-decisions-2026-09-27).
Resolved planning choices below do not mark M1 phases complete.

- [x] **Q1 species alignment:** D3 chooses explicit process-owned lane mapping,
  full gas-order environment lanes and preservation of nonparticipating material.
  Nonempty unique gas names and expected configuration order are validated;
  unnamed particle chemistry remains caller-declared, not inferred from shape.
- [x] **Q2 concentration convention:** D1 chooses counts per simulation volume
  for discrete/resolved and radius PDF dN/dr for continuous_pdf. Physical
  concentration divides by V once; PDF populations integrate over radius in m.
  Shared distribution_type metadata uses the existing three-value vocabulary.
  D2 explicitly authorizes necessary CPU/GPU correction and metadata work.
- [x] **Q3 access API:** D4 chooses particles/gas/environment properties with
  validated setters and all-three replace_data; copies detach, raw fields are
  writable, derived properties remain fresh. No redundant get_*/set_* family.
- [x] **Q4 helpers/validation placement:** reuse existing derived properties;
  provide consumer-backed normalization/population helpers. Proposed concrete
  aerosol_validation.py owns shared structural checks; process admission owns
  physics, mapping and distribution compatibility. No reconstruction to validate.
- [x] **Q5 governance:** Kyle/Gorkowski approves semantics and final M1 evidence.

## Evidence still required in M1

P1 freezes exact helper signatures, PDF grid/quadrature, metadata-construction
policy and the complete CPU/GPU consumer ledger against D1–D3. Characterize
V=0.25/1/4 and unequal supported weights independently; do not bless legacy
agreement as correctness. P2 publishes the D4 replacement acceptance matrix.
P3/P4 implement bounded helpers/metadata admission with adjacent tests and
transfer implications; P5 supplies literal final-revision results and approval.
No normalization implementation or passing characterization is claimed here.
