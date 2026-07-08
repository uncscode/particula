## Architecture Design

### High-Level Design

```text
docs example source (.py, Jupytext percent style)
  -> import particula as par
  -> build GasSpecies / Atmosphere / ParticleRepresentation / Aerosol
  -> LatentHeatFactory creates constant latent heat strategy
  -> CondensationFactory creates CondensationLatentHeat strategy
  -> MassCondensation executes CPU condensation step(s)
  -> example reads strategy.last_latent_heat_energy
  -> print/plot per-step and cumulative energy diagnostics
  -> optional notebook sync mirrors the source example
```

The example is documentation-layer code only. It exercises existing CPU
condensation APIs and does not introduce a new runtime abstraction.

### Data / API / Workflow Changes

- **Data Model:** No production data-model changes. The example records local
  arrays/lists for time, particle mass, gas concentration, per-step energy, and
  cumulative energy.
- **API Surface:** No new public APIs are planned. The feature should use
  existing public exports from `particula.dynamics` and `particula.gas`.
- **Workflow Hooks:** Documentation workflow should edit `.py` first, lint it,
  sync `.ipynb` if paired, execute the notebook, and update the examples index.

### Example Behavior Requirements

- Construct a latent heat strategy, not just a vapor-pressure strategy.
- Run `par.dynamics.MassCondensation(...).execute(...)` with a real aerosol.
- Read `condensation_strategy.last_latent_heat_energy` after each execution
  step and describe sign convention: positive for condensation, negative for
  evaporation.
- Make clear that the value is per-step energy bookkeeping in joules and is not
  a temperature-feedback solver.

### Security & Compliance

No new permissions, network calls, credentials, or platform integrations are
introduced. The example must remain deterministic and CPU-only. It should not
attempt GPU device discovery or make unsupported GPU parity statements.
