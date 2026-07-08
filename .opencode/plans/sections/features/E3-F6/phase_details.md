## Phase Details

- [ ] **E3-F6-P1:** Draft runnable CPU CondensationLatentHeat example with energy bookkeeping
  - Issue: TBD | Size: S | Status: Not Started
  - Depends on: Existing CPU condensation builders/factories remaining the
    source of truth; do not add notebook/index wiring before the base example is
    runnable.
  - Goal: Add a runnable example that constructs latent heat through public
    factories/builders, runs `MassCondensation.execute()`, and reports actual
    `last_latent_heat_energy` values from condensation mass transfer.
  - Files: `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py`
    and supporting docs comments within the example; add
    `particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py`
    if no adjacent example smoke test already fits.
  - Tests: Run the script directly and add a smoke test that imports or executes
    the example entrypoint, asserts it completes on CPU-only environments, and
    verifies a finite non-zero latent-heat energy diagnostic. Lint/format the
    example source, and include any small co-located regression test only if a
    production API fix is required.

- [ ] **E3-F6-P2:** Pair notebook and index documentation for the latent-heat example
  - Issue: TBD | Size: S | Status: Not Started
  - Depends on: E3-F6-P1 landing the final example path and narrative so notebook
    sync and index links do not target a moving artifact.
  - Goal: Sync the example to a notebook if following the existing docs pattern
    and make it discoverable from the Dynamics examples index.
  - Files:
    `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb`,
    `docs/Examples/Dynamics/index.md`, and optional targeted cross-link in
    `docs/Features/condensation_strategy_system.md`.
  - Tests: Validate notebook sync and execute the notebook when paired; verify
    markdown links touched by the phase; extend the same example smoke test to
    confirm the documented example path and index link stay aligned.

- [ ] **E3-F6-P3:** Validate example execution, notebook sync, and CPU-only guidance
  - Issue: TBD | Size: XS | Status: Not Started
  - Depends on: E3-F6-P1 and E3-F6-P2 producing the final runnable example and
    any paired notebook artifacts.
  - Goal: Run final documentation validation and confirm the example demonstrates
    real CPU latent-heat bookkeeping without GPU parity claims.
  - Files: Validation notes in docs or plan updates as needed.
  - Tests: Re-run
    `particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py`,
    run the example, notebook execution if paired, `ruff check`/`ruff format`
    on the example, and docs validation where available. If builder/factory or
    strategy APIs changed during the example work, also re-run the focused
    condensation regression modules named in `testing_strategy.md`.
