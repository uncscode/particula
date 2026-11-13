# Open Questions and Review Decisions

Reviewed with Kyle Gorkowski on 2026-09-27. The canonical decision and research
record is [appendix D1–D5](appendix.md#review-decisions-2026-09-27).
Checked entries mean a planning decision was made, not implementation shipped.

- [x] **Species mapping:** explicit process-owned gas-to-particle lane map;
  environment follows full gas order. Preserve nonpartitioning gas and
  particle-only material. Structural validation does not infer chemistry.
- [x] **API and mutation:** keyword-only `Aerosol(particles=..., gas=...,
  environment=...)`, identity-returning properties/validated setters and atomic
  all-three `replace_data(...)`. Raw fields are writable; derived values are
  fresh; copies detach data and preserve metadata.
- [x] **Initialization:** retain direct mass/radius, speciated, PDF/PMF and
  sampled-lognormal capabilities through native builders/utilities. Preserve
  established defaults/RNG unless an explicit change is approved.
- [x] **Transition:** retain narrowly isolated legacy paths while native paths
  migrate; no facade emulation on the flat aggregate or duplicate state.
  Concrete consumer/aliasing/deletion ledger is required at M2 entry.
- [x] **Examples/history:** retain all supported scientific content and execute
  full published workloads. No historical exceptions approved; M5 expands the
  existing seed inventory to every source/pair/snippet and records resources.
- [x] **Governance:** Kyle/Gorkowski is final plan/scientific/release approver.
  No fixed deadline. Use `data-native-v030-closeout.md` as the evidence index;
  actual execution operators and revisions are recorded by owning tracks.
- [x] **Plan representation:** retain schema-supported epic milestones and
  maintenance-child phases. Fixed M3 metadata to depend on E9-M2.
- [x] **Normalization correction:** raw counts per simulation volume for
  discrete/resolved; radius-based `dN/dr` per simulation volume for PDFs.
  Divide by V exactly once for physical concentration. Shared distribution_type
  vocabulary on data and processes replaces ambiguous storage interpretation.
  Radius in metres is the sole distribution coordinate, confirmed by maintainer.
- [x] **Scope amendment:** include required CPU/GPU normalization and metadata
  propagation, including prepared/resident/checkpoint consequences. Preserve
  array layouts, device ownership, explicit transfer APIs and no implicit rebind.

## Remaining implementation gates

M1 must deliver the full consumer/unit/metadata ledger and independent
non-unit-volume/PDF characterization; M2 must validate its concrete transition;
M3/M4 must prove scientific and cross-backend corrections; M5 must complete its
path-level inventory and execution evidence; M6 must prove removal and readiness.
These are assigned phase deliverables, not further undecided API alternatives.
No implementation gate or scientific test is claimed complete by this review.
