# Dependencies

Parent: E9, v0.3.0 data-native Aerosol migration, issue #1602.
T1..T6 map exactly to E9-M1..E9-M6; all are maintenance tracks.

| Plan | Relationship and gate |
|---|---|
| E9-M1 / T1 | Authority for species mapping, units, normalization, helpers and ownership |
| E9-M2 / T2 | Hard prerequisite: native construction, flat aggregate and replacement completed and validated before M3 starts |
| E9-M3 / T3 | Native CPU science and enduring scientific regression fixtures |
| E9-M4 / T4 | Runnables, dilution/nucleation orchestration and CPU adapters; starts after M3 completed and validated |
| E9-M5 / T5 | Examples, notebooks and broad documentation after M4 validation |
| E9-M6 / T6 | Legacy removal and release readiness last, after M5 validation |

Strict **M1 → M2 → M3 → M4 → M5 → M6**, no parallel implementation.
Planning is not completion evidence. Internally P1→P2→P3→P4→P5→P6 also
requires validated gates before progression. Upstream unresolved contracts
block implementation rather than justify a competing M3 convention.

Use existing NumPy/SciPy, pytest, Ruff/mypy and docs tooling; no new external
dependencies. GPU conversion/export/ownership/resident/capture contracts are
protected, not redesign targets. Epic I follows E9, not a prerequisite.

This draft changes only E9-M3 phases and canonical sections. Dependency
semantics are recorded here; sibling metadata/contracts remain untouched.
