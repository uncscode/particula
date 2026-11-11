# Phase Details

## Current implementation and serial evidence

The approved M1 handoff and merged PR #1624 preceded M2. All six phases are
implemented on stacked branches in P1 → P2 → P3 → P4 → P5 → P6 order. Each
implementation phase began after its predecessor's local validation passed.
PR merge/review and final maintainer M3 admission are separate gates; no merge
or M3 implementation is claimed. Final-revision evidence is in
`validation_evidence.md`.

| Phase | Delivered behavior | PR / evidence |
|---|---|---|
| P1 | Keyword-only three-container aggregate; identity reads; structural then representation admission; legacy isolation | #1625, `p1_working_evidence.md` |
| P2 | All setters and atomic whole-triple replacement; same-object revalidation and rejection snapshots | #1626, `p2_validation.md` |
| P3 | Native fluent builder, constructor-consistent admission and repeated-build identity; temporary legacy builder | #1627, `p3_validation.md` |
| P4 | Owned single-material radius and lognormal PDF/PMF native initialization | #1628, `p4_validation.md` |
| P5 | Owned direct resolved mass/count/density and sampled-lognormal initialization | #1629, `p5_validation.md` |
| P6 | Bounded developer guide, data-only integration, reconciled ledgers and final validation | #1630, `validation_evidence.md` |

## Contract applied to every phase

Native builders/initializers declare kind explicitly. Discrete/PMF densities
and radius PDF dn/dr are already per-volume and use fixed volume 1 m³. Only
resolved density-to-count conversion multiplies by physical volume; direct
counts do not. Non-unit-volume success cases belong to resolved storage;
non-unit binned volumes reject. PDF mass density integrates mass times PDF.

Aerosol validates structure then calls ParticleData.validate_representation;
it never copies/coerces candidates, stores process mappings, or rebinds GPU
state. Chemical mapping and unnamed gas/environment order are not certified by
structural width checks. Native tagged Warp upload remains unsupported and
rejects rather than silently dropping metadata.

Each phase includes its co-located behavioral tests. The capability and exact
legacy-consumer ledgers give successor assertions and M3/M4/M5/M6 ownership.
M2 completion requires final validation and explicit maintainer handoff
acceptance before M3 starts.
