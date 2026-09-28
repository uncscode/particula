# Open Questions and Review Decisions

Authority: [E9 review decisions](../../epics/E9/appendix.md#review-decisions-2026-09-27).

- [x] **Q1 contract authority:** consume approved D1–D4 and M1's eventual
  completed-and-validated implementation packet. Actual revision hashes are
  handoff evidence, not a second API design choice.
- [x] **Q2 transition policy:** preserve isolated legacy paths until migration;
  if needed isolate the old aggregate in a temporary concrete module rather
  than add atmosphere/facade emulation to native Aerosol. Exact consumers,
  aliasing/mutation rules and M6 deletion proof must be reviewed before P1.
- [x] **Q3 retained construction:** reuse native builders, migrate AerosolBuilder
  and retain direct mass/radius, speciated, radius PDF/PMF and sampled-lognormal
  capabilities in small native utilities. Every result declares distribution_type.
- [x] **Q4 defaults/RNG:** preserve existing sampling/default behavior unless
  separately approved. Native density inputs explicitly multiply by volume;
  direct counts do not. Sampling modal weights does not automatically make a
  sample count equal a prescribed physical number density. Document both.
- [x] **Q5 governance:** Kyle/Gorkowski accepts M1 entry and M2 completion.

## Evidence still required in M2

P1 freezes the concrete transition ledger and demonstrates retained-consumer
smoke/full-suite behavior. P3–P5 expand the capability ledger and trace every
retained assertion/default/unit conversion to its native successor. Characterize
sampling with controlled fixtures and the existing random-state implementation;
no new cross-backend RNG equivalence. P6 records final revision, commands,
results and approval. API approval alone does not authorize M3 admission.
