# Open Questions and Review Decisions

Authority: [E9 review decisions](../../epics/E9/appendix.md#review-decisions-2026-09-27).

- [x] **Q1 upstream contract:** consume D1–D4 and completed M1–M3 evidence;
  accessor/mapping/normalization choices are settled. Revisions remain entry
  evidence to be produced by implementation.
- [x] **Q2 environment:** remove Nucleation's separate environment argument from
  the final API; read current aerosol.environment on execution. CPU replacement
  must be observed without automatic prepared GPU/resident rebinding.
- [x] **Q3 nucleation admission:** replace facade type tests with native
  distribution compatibility, fixed-slot, mapping and physical checks. Preserve
  existing demonstrated topology; reject PDFs/unsupported weighted behavior
  rather than widening scientific support merely because a class disappears.
- [x] **Q4 compatibility:** retain only inventoried M2/M3 transition consumers,
  including custom-runnable fixtures, with M6 deletion ownership.
- [x] **Q5 governance:** Kyle/Gorkowski accepts entry/completion, with literal
  final-revision evidence linked from the appendix and E9 closeout index.

## Accepted implementation decisions and evidence

The implementation record covers D1–D2 consequences for CPU/GPU dilution and
nucleation, resampling,
volume evolution, communication, resident composition, metadata compatibility
and checkpoint restore. Distinguish physical expansion (fixed counts) from
representative-volume scaling (counts and volume scaled together). Version or
reject ambiguous old checkpoint semantics explicitly. Retain current-gas
substeps and per-attempted-substep atomicity. See
[implementation evidence](implementation.md) for exact commands and results.
Final maintainer acceptance and M5 authorization remain pending.

The maintainer accepted single-PR delivery, canonical free-slot wall-loss
retirement, explicit native source mapping/current environment, and rejection
of old checkpoint formats. No schema migration utility is included.
