# E2-F8 Infrastructure Reuse

## Existing Code to Reuse

- `particula/particles/particle_data.py`
  - Defines `ParticleData` with `(n_boxes, n_particles, ...)` arrays and
    single-box simulations represented as `n_boxes=1`.
- `particula/gas/gas_data.py`
  - Defines `GasData` with `(n_boxes, n_species)` concentrations.
- `particula/dynamics/condensation/condensation_strategies.py`
  - Reuse `_unwrap_particle`, `_unwrap_gas`, `_require_matching_types`, and
    `_require_single_box` for explicit single-box validation.
- `particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py`
  - Existing `ParticleData` adapters (`_get_radius`, `_get_mass`,
    `_get_concentration`, `_get_volume`) currently read box 0; use this as the
    audit target and likely insertion point for validation.
- Existing tests:
  - `particula/dynamics/condensation/tests/condensation_strategies_test.py`
  - `particula/dynamics/coagulation/coagulation_strategy/tests/coagulation_strategy_abc_test.py`

## Existing Documentation to Reuse

- `docs/Features/particle-data-migration.md`
  - Already documents dynamics accepting data containers and has a
    "Single-box vs multi-box data" section; expand it with a support table.
- `docs/Features/Roadmap/data-oriented-gpu.md`
  - Use for roadmap-level language that container support does not imply all
    strategies execute all boxes.

## Patterns to Preserve

- Google-style docstrings and clear `ValueError`/`TypeError` messages.
- `*_test.py` co-located test naming.
- Single-box legacy compatibility using box 0 where explicitly documented.
- Small, reviewable phases with tests shipping alongside implementation.

## Gaps to Close

- Condensation has helper-level single-box tests, but public strategy methods
  need clearer multi-box rejection coverage.
- Coagulation lacks an explicit multi-box boundary test and may silently mutate
  only box 0 for multi-box `ParticleData`.
- Docs currently state dynamics accept containers without enough strategy-level
  limitations.
