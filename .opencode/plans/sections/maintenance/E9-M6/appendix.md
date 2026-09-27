# Appendix: Removal and Evidence Ledgers

This is an audit seed, not an approved deletion list or execution record.

| Inventory family | Concrete seed | Required disposition |
|---|---|---|
| Particle facade | particles/representation.py, representation_builders.py, representation_factories.py, change_particle_representation.py | Remove obsolete facade surfaces; retain scientific algorithms/native initialization |
| Particle fixtures | particles/tests/representation{,_facade,_builders,_factories,_zero_particle}_test.py; change_particle_representation_test.py | Map enduring numerical assertions before deleting facade mechanics |
| Gas/environment facade | gas/species.py, species_builders.py, species_factories.py, atmosphere.py, atmosphere_builders.py | Remove obsolete surfaces; preserve native gas/environment and thermodynamics |
| Gas fixtures | gas/tests/species{,_private,_facade,_builder,_factory}_test.py; atmosphere{,_builder}_test.py | Classify assertions individually, not by legacy filename |
| Aggregate/consumer seams | aerosol.py, aerosol_builder.py, runnable.py, dynamics/, execution/adapters/ | Close all M1–M5 temporary compatibility rows |
| Protected transfers | gpu/conversion.py, gpu/tests/conversion_test.py | Retain CPU-Warp helpers and explicit ownership |
| Protected resident contracts | execution/tests/, gpu/kernels/tests/ | Retain imports, lifecycle, device, RNG and failure boundaries |
| Published workflows | M5 appendix inventory; docs/Examples/, migration pages, root/developer snippets | Execute supported workflows; narrow explicit historical exceptions |

Code paths in the table are relative to `particula/` unless prefixed `docs/`.
Search seeds include `ParticleRepresentation`, `GasSpecies`, `Atmosphere`,
their concrete module paths, builder/factory names, `.atmosphere`, aliases and
temporary seam names from M1–M5. Text search is necessary but insufficient:
review imports, dynamic dispatch, annotations, fixtures and generated outputs.
Do not mistake legitimate atmospheric science vocabulary for a deleted API.

## Required implementation records

1. Consumer/seam: path + symbol + use; upstream owner/revision; replacement;
   removal phase; test/oracle mapping; final search result; reviewer.
2. Test retention: old node/assertion; physical invariant; new native node;
   units/shape/volume/species cases; tolerance; result; deletion justification
   for facade-only rows. No empty replacement for enduring scientific coverage.
3. Historical exception: exact path/section; version label; why retained;
   confirmation non-executable and not supported API guidance; approver.
4. Validation: date/revision; environment/dependencies/devices; literal command;
   output/artifact pointer; pass/fail/skip/unavailable; mandatory/optional;
   blocker owner and rerun. Keep focused assertion and full coverage separate.
5. Examples: consume every M5 row, including unchanged supported examples,
   notebook pair paths, execution outputs and approved historical exceptions.

P6 may publish these ledgers in the proposed closeout record; until then keep
them with owning PR evidence. Initial execution status: NOT RUN / readiness
BLOCKED. No historical GPU timing/profiling gap is resolved by this maintenance.

References: issue #1602; E9 `outcomes_guardrails` and `dependency_map`;
E9-M1 `guidelines_requirements`; E9-M2 `success_criteria`; E9-M3
`testing_requirements`; E9-M4 `scope`; E9-M5 `appendix` and `dependencies`;
`.opencode/guides/testing_guide.md`; `pyproject.toml`.
