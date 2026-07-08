# Phase Details

- [ ] **E3-F7-P1:** Adapt particle-resolved latent-heat fixture as an integration test
  - Issue: TBD | Size: S | Status: Not Started
  - Depends on: Existing CPU latent-heat APIs staying stable and, when
    available, E3-F6 providing the finalized example path for later cross-links;
    the integration fixture itself does not need to wait for E3-F6 to merge.
  - Goal: Create a minimal deterministic single-species CPU integration fixture
    based on the existing particle-resolved condensation style, but using
    `CondensationLatentHeat`.
  - Files: `particula/integration_tests/condensation_latent_heat_conservation_test.py`,
    optionally shared local helpers within the same test module.
  - Tests: New integration test constructs an aerosol, partitioning gas species,
    constant latent-heat strategy, and `MassCondensation` runnable through CPU
    public APIs.
  - Extension: Add a small multi-species variant only if it remains fast,
    deterministic, and stable under the default integration suite.

- [ ] **E3-F7-P2:** Assert CPU mass conservation and latent-heat energy bookkeeping
  - Issue: TBD | Size: S | Status: Not Started
  - Depends on: E3-F7-P1 establishing the deterministic fixture so conservation
    and latent-heat assertions target a stable baseline instead of evolving setup
    code.
  - Goal: Add robust assertions proving particle/gas water inventory conservation
    and latent-heat energy consistency for the CPU reference path.
  - Files: `particula/integration_tests/condensation_latent_heat_conservation_test.py`,
    with minimal updates to condensation helpers only if required for clarity.
  - Tests: Assert particle water increases, gas water decreases, total water mass
    is conserved within stable tolerance, `last_latent_heat_energy` is finite and
    positive, and energy equals transferred mass times constant latent heat.

- [ ] **E3-F7-P3:** Document Epic D CPU latent-heat baseline and validation guidance
  - Issue: TBD | Size: XS | Status: Not Started
  - Depends on: E3-F7-P2 proving the executable baseline and, if E3-F6 has
    landed, using that final example path for documentation cross-links without
    blocking the CPU integration baseline on docs-example timing.
  - Goal: Document the new integration baseline as CPU-only reference evidence
    for future Epic D GPU parity work.
  - Files: `docs/Features/Roadmap/data-oriented-gpu.md`,
    `docs/Features/condensation_strategy_system.md`.
  - Tests: Documentation changes are reviewed for accurate CPU-only wording and
    the default integration test remains the executable validation artifact.
  - Deliverable: Update developer-facing roadmap or feature docs so the final
    phase leaves explicit CPU-reference guidance for later GPU parity work.
