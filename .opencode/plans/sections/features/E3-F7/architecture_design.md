# Architecture Design

## Design Summary

E3-F7 should add a narrow CPU integration baseline rather than new production
architecture. The test drives the same public runnable path a user would invoke:

1. construct a particle-resolved aerosol with a partitioning water gas species;
2. construct `CondensationLatentHeat` with a deterministic constant latent heat;
3. wrap it in `MassCondensation`;
4. execute a short fixed sequence of CPU time steps; and
5. compare before/after particle, gas, and latent-heat diagnostics.

## Data Flow

```text
Aerosol(particles, gas, atmosphere)
    -> MassCondensation(CondensationLatentHeat).execute(...)
    -> updated particle water mass
    -> updated gas water concentration
    -> CondensationLatentHeat.last_latent_heat_energy
```

The conservation assertion should compute initial and final water inventory from
particle water mass concentration plus gas water concentration in the same style
as the existing particle-resolved condensation integration test. The latent-heat
assertion should derive transferred mass from the gas/particle inventory delta
and compare against `latent_heat_ref * transferred_mass`.

## Boundaries

- The test must stay in `particula/integration_tests/` rather than GPU test
  modules.
- No Warp imports, CUDA checks, or GPU data containers should appear in this
  feature.
- Any helper extraction should be local to the test unless multiple existing
  integration tests clearly benefit.

## Compatibility

The design relies on existing CPU APIs and should not change user-facing
behavior. If small tolerance adjustments are needed, prefer making the fixture
more deterministic over weakening conservation expectations.
