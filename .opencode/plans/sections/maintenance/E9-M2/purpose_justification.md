# Purpose and Justification

E9-M2 implements T2 of issue #1602 under parent E9: data-native construction
and a flat `Aerosol` holding `ParticleData`, `GasData`, and `EnvironmentData`.
It consumes the reviewed E9-M1 units, ordered alignment, access and replacement
contracts; it does not establish a competing contract authority.

The current aggregate (`particula/aerosol.py`) holds an `Atmosphere` and a
`ParticleRepresentation`, and its replacement methods assign without a joint
compatibility check. `AerosolBuilder` depends on facade strategy names and
partitioning-species counts. Useful radius-bin and resolved initialization
also constructs facades. Leaving these paths unchanged prevents the later
scientific/runnable migration and makes v0.3.0 facade removal unsafe.

The maintenance outcome is practical native construction, preserved supplied
container identity, and validated all-or-none replacement. Independent
non-unit-volume and ordered-species tests protect against lost normalization,
silent gas omission and partial publication. Physics remains process-owned;
there is no replacement compatibility facade or duplicated authoritative data.

Implementation starts only after E9-M1 is completed and validated. M2's
completed-and-validated handoff admits M3; M4 retains runnable migration,
M5 broad examples/docs, and M6 removal. This is a Draft maintenance plan,
not implementation evidence, release approval, or a diagnostics track.
