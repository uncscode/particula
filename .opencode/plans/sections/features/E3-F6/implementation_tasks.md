## Implementation Tasks

### E3-F6-P1: Runnable Example Source

- Create `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py` in
  existing Jupytext percent format.
- Add markdown cells explaining learning objectives, CPU-only scope, and energy
  bookkeeping semantics.
- Build a minimal but physical aerosol/gas scenario using existing documented
  constructors.
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

### E3-F6-P2: Notebook and Discoverability

- Sync the `.py` source to
  `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb` if using
  the paired notebook convention.
- Execute the notebook so outputs are valid for docs publication.
- Add a bullet/link in `docs/Examples/Dynamics/index.md`.
- Add a targeted cross-link from `docs/Features/condensation_strategy_system.md`
  if needed so users can find the full example from the feature page.

### E3-F6-P3: Final Validation and Guardrails

- Run `python docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py`.
- Run ruff check/format on the example source.
- Run notebook sync/execution commands if an `.ipynb` is present.
- Confirm no `particula.gpu`, Warp, CUDA, or GPU parity statements appear in
  the example or touched docs.
- If implementation required production fixes, run focused condensation tests
  and document the reason in the change log.
