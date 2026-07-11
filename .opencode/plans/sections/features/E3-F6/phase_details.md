## Phase Details

- [x] **E3-F6-P1:** Draft runnable CPU CondensationLatentHeat example with energy bookkeeping
  - Issue: #1263 | Size: S | Status: Shipped in issue scope
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
  - Tests: The shipped work adds
    `particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py`
    for smoke/invariant coverage around the example entrypoint and helper
    results. Lint/format and direct script execution remain the core validation
    path.

- [ ] **E3-F6-P2:** Pair notebook and index documentation for the latent-heat example
  - Issue: follow-up TBD | Size: S | Status: Deferred; not shipped in issue #1263
  - Depends on: E3-F6-P1 landing the final example path and narrative so notebook
    sync and index links do not target a moving artifact.
  - Goal: Sync the example to a notebook if following the existing docs pattern
    and make it discoverable from the Dynamics examples index.
  - Files:
    `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb`,
    `docs/Examples/Dynamics/index.md`, and optional targeted cross-link in
    `docs/Features/condensation_strategy_system.md`.
  - Tests: Still pending because no `.ipynb` artifact or index wiring was added
    in the shipped implementation.

- [ ] **E3-F6-P3:** Validate example execution, notebook sync, and CPU-only guidance
  - Issue: follow-up TBD | Size: XS | Status: Partially satisfied for `.py` validation only
  - Depends on: E3-F6-P1 and E3-F6-P2 producing the final runnable example and
    any paired notebook artifacts.
  - Goal: Run final documentation validation and confirm the example demonstrates
    real CPU latent-heat bookkeeping without GPU parity claims. For issue #1263,
    this was satisfied only for the runnable `.py` artifact and focused tests.
  - Files: Validation notes in docs or plan updates as needed.
  - Tests: Re-run
    `particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py`,
    run the example, and `ruff check`/`ruff format` on the example. Notebook
    execution remains deferred until a paired artifact exists. No additional
    production regression module was required by the shipped implementation.
