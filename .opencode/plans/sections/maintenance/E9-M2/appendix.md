# Appendix

## Evidence and references

| Source | Planning use |
|---|---|
| Issue #1602 via ADW issue state | T2 ownership and agreed invariants |
| E9 implementation_strategy/dependency_map/open_questions | Serial gates and temporary compatibility review |
| E9-M1 guidelines_requirements/phase_details/testing_requirements | Upstream replacement specification, normalization/alignment authority and validation policy |
| `particula/aerosol.py:37–96` | Existing facade constructor and unchecked replacements |
| `particula/aerosol_builder.py:107–173` | Legacy strategy/count validation and aggregate build |
| `particula/particles/representation_builders.py:417–461,575–636` | PDF/PMF and sampled resolved initialization capabilities |
| `particula/particles/particle_data_builder.py` | Reusable native construction surface |
| `.opencode/guides/testing_guide.md`, `pyproject.toml` | Focused/full-suite split, local warning policy and wall-loss exclusion |
| Maintenance templates and E9-M1 draft | Canonical format and relevant maintenance prior art, not completed implementation evidence |

## Seed capability ledger (review/finalize before implementation)

| Existing capability | Proposed disposition | Phase | Removal boundary |
|---|---|---|---|
| Aerosol / AerosolBuilder facade inputs | Native three-container path | P1/P3 | Any temporary legacy path: M6 |
| Native particle/gas data builders | Reuse; add only approved gaps | P3 | Retained, not obsolete bridges |
| Mass/radius representation construction | Native data inputs/conversion | P3/P4 | Legacy builder names: M6 |
| PresetParticleRadiusBuilder PDF/PMF | Retain useful distribution generation | P4 | Facade-bearing preset: M6 |
| Resolved mass / sampled preset | Native data initialization | P5 | Facade-bearing builder: M6 |
| CPU↔Warp conversion helpers | Preserve explicit transfer API/raw arrays; propagate D1 distribution metadata | P6 checks | Retained, never M6 deletion targets |

For every temporary compatibility seam, record exact path/symbol, reason,
consumer list, held-state authority/aliasing rules, entry phase, tests,
migration owner (M3 scientific, M4 runnable/adapter, M5 examples as applicable),
and **E9-M6 removal owner/gate**. None is approved or implemented by this draft.
Do not close M2 with an unbounded "compatibility layer" entry. M6 must remove
the seam only after all named consumers migrate and enduring tests survive.

Behavioral-test ledger rows pair old node/fixture with new native node, units,
oracle, non-unit-volume case and owning PR. Validation rows record revision,
date, literal command, result/counts, availability, evidence path and reviewer.
These implementation ledgers remain pending; no passing runs are fabricated.

## Drafting limitations

Only E9-M2 metadata/sections are edited. Canonical returned relative paths match
the maintenance/E9-M2 prefix with no traversal; access uses the exact worktree.
Full-file `apply_patch` replacements substitute for unavailable `write` tooling.
No independent lstat/symlink-audit tool is available. No schema command or
implementation test execution is performed in this planning-only task.
