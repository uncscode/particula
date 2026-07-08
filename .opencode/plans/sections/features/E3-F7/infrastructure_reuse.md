# Infrastructure Reuse

## Existing Test Pattern

- `particula/integration_tests/condensation_particle_resolved_test.py` provides
  the preferred deterministic particle-resolved integration style. Reuse its
  setup pattern for vapor pressure, partitioning gas species, atmosphere,
  particle-resolved representation, short time stepping, and conservation
  assertions.

## CPU Latent-Heat Implementation

- `particula/dynamics/condensation/condensation_strategies.py` contains
  `CondensationLatentHeat`, including the `last_latent_heat_energy` diagnostic.
- `particula/dynamics/particle_process.py` contains `MassCondensation.execute()`,
  the runnable integration wrapper that should drive the baseline.
- `particula/dynamics/condensation/mass_transfer.py` contains
  `get_latent_heat_energy_released(mass_transfer, latent_heat)`, the helper
  behavior that defines `Q = dm * L`.

## Builders, Factories, and Public Exports

- `particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py`
  documents the builder path if the test prefers builder construction.
- `particula/dynamics/condensation/condensation_factories.py` supports
  `strategy_type="latent_heat"` if factory coverage is useful.
- `particula/dynamics/__init__.py` and `particula/dynamics/condensation/__init__.py`
  export latent-heat condensation APIs for public-path tests.
- `particula/gas/latent_heat_strategies.py` provides
  `ConstantLatentHeat(latent_heat_ref=...)` for deterministic bookkeeping.

## Documentation Targets

- `docs/Features/Roadmap/data-oriented-gpu.md` already identifies the need for
  E3-F7 as an Epic D CPU baseline.
- `docs/Features/condensation_strategy_system.md` is the existing latent-heat
  feature documentation target.
- E3-F6 example material can be cross-linked when present.
