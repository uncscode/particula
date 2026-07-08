# Phase Details

- [ ] **E3-F7-P1:** Adapt particle-resolved latent-heat integration fixture with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Create a deterministic CPU integration fixture based on the existing
    particle-resolved condensation style, but using `CondensationLatentHeat`.
  - Files: `particula/integration_tests/condensation_latent_heat_conservation_test.py`,
    optionally shared local helpers within the same test module.
  - Tests: New integration test constructs an aerosol, partitioning gas species,
    constant latent-heat strategy, and `MassCondensation` runnable through CPU
    public APIs.

- [ ] **E3-F7-P2:** Assert CPU mass conservation and latent-heat energy bookkeeping
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Add robust assertions proving particle/gas water inventory conservation
    and latent-heat energy consistency for the CPU reference path.
  - Files: `particula/integration_tests/condensation_latent_heat_conservation_test.py`,
    with minimal updates to condensation helpers only if required for clarity.
  - Tests: Assert particle water increases, gas water decreases, total water mass
    is conserved within stable tolerance, `last_latent_heat_energy` is finite and
    positive, and energy equals transferred mass times constant latent heat.

- [ ] **E3-F7-P3:** Document Epic D CPU latent-heat baseline and validation guidance
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document the new integration baseline as CPU-only reference evidence
    for future Epic D GPU parity work.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/condensation_strategy_system.md`.
  - Tests: Documentation changes are reviewed for accurate CPU-only wording and
    the default integration test remains the executable validation artifact.
