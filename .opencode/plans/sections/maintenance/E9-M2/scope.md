# Scope

## Modules and directories

- `particula/aerosol.py`, `particula/aerosol_builder.py` and adjacent
  `particula/tests/aerosol_test.py` / `aerosol_builder_test.py`.
- Existing `particula/particles/particle_data_builder.py` and
  `particula/gas/gas_data_builder.py`, with adjacent tests: reuse first;
  modify only construction gaps approved by the M1/M2 inventory.
- Useful capabilities in `particula/particles/representation_builders.py`
  and its tests: mass/radius input, lognormal radius-bin PDF/PMF initialization,
  resolved mass initialization and sampled resolved presets. Provide native
  equivalents in reviewed concrete construction modules, not new state facades.
- Minimal construction exports and docstrings, a bounded developer reference
  in `docs/Features/particle-data-migration/`, and data-only integration tests.

## Interfaces and acceptance boundary

Flat constructor, identity-preserving whole-container access, individual
replacement, coordinated three-container replacement, native aggregate builder
and retained useful initialization. Final public spelling and generic schema
versus process validation are consumed from M1, not guessed here. Preserve all
gas species with ordered names/partitioning metadata and environment alignment.

Construction behavioral tests migrate in the same PR as their implementation.
Inventory any strictly necessary temporary seam for unmigrated consumers:
path, concrete consumers, authority/aliasing rules, introducing phase, migration
owner and M6 removal gate. Approval is required before introducing a seam.

## Out of scope

- M1 contract/helper redesign; escalate missing or contradictory prerequisites.
- M3 scientific condensation/coagulation/wall-loss migration or new physics.
- M4 `Runnable` composition, dilution/nucleation orchestration, CPU adapters.
  Preserve their existing behavior during M2; do not migrate them early.
- M5 supported tutorial/notebook migration; M6 deletion of facades,
  `Atmosphere`, obsolete builders/exports/bridges/messages and facade-only tests.
- Storage/precision redesign, general multi-box CPU execution, new GPU public
  APIs, hidden transfer/fallback, automatic GPU/resident rebinding, capture
  redesign, benchmarks, diagnostics or Epic I.

This drafting operation edits only E9-M2 metadata and its twelve section files;
all production paths above are future implementation scope.
