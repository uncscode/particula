## Implementation Tasks

### E3-F6-P1: Runnable Example Source

- Create `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py` in
  existing Jupytext percent format.
- Add markdown cells explaining learning objectives, CPU-only scope, and energy
  bookkeeping semantics.
- Build a minimal but physical aerosol/gas scenario using public builders that
  already appear in CPU condensation examples, such as `GasSpeciesBuilder`,
  `AtmosphereBuilder`, and
  `ResolvedParticleMassRepresentationBuilder`, so the example stays copyable.
- Create a constant latent heat strategy with `par.gas.LatentHeatFactory()`.
- Create a `CondensationLatentHeat` strategy via
  `par.dynamics.CondensationFactory().get_strategy("latent_heat", ...)` or the
  public builder.
- Wrap it with `par.dynamics.MassCondensation` and execute multiple short CPU
  timesteps.
- Collect and display gas concentration, particle mass, per-step latent heat
  energy, and cumulative energy.
- Add comments that distinguish latent-heat energy bookkeeping from temperature
  feedback.
- Keep the executable path reviewable: aim for one notebook-style file with a
  small setup block, one timestep loop, and one reporting block rather than a
  large reusable helper framework.

### E3-F6-P2: Notebook and Discoverability

- Sync the `.py` source to
  `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb` if using
  the paired notebook convention.
- Execute the notebook so outputs are valid for docs publication.
- Add a bullet/link in `docs/Examples/Dynamics/index.md`.
- Add a targeted cross-link from `docs/Features/condensation_strategy_system.md`
  if needed so users can find the full example from the feature page.
- Keep the discoverability edit bounded to those two docs locations unless a
  third link is required to prevent an orphaned example.

### E3-F6-P3: Final Validation and Guardrails

- Run `python docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py`.
- Run ruff check/format on the example source.
- Run notebook sync/execution commands if an `.ipynb` is present.
- Confirm no `particula.gpu`, Warp, CUDA, or GPU parity statements appear in
  the example or touched docs.
- If implementation required production fixes, run focused condensation tests
  and document the reason in the change log.
- If extra validation code is needed, keep it in the example file or the
  existing docs validation flow rather than creating a separate latent-heat-only
  helper module.
