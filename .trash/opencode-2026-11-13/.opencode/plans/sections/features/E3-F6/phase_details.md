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

- [x] **E3-F6-P2:** Pair notebook and index documentation for the latent-heat example
  - Issue: #1264 | Size: S | Status: Shipped
  - Depends on: E3-F6-P1 landing the final example path and narrative so notebook
    sync and index links do not target a moving artifact.
  - Goal: Publish the paired notebook artifact, make it discoverable from the
    Dynamics examples index, and add one bounded feature-doc cross-link when
    helpful.
  - Files:
    `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb`,
    `docs/Examples/Dynamics/index.md`,
    `docs/Features/condensation_strategy_system.md`, and focused docs-surface
    assertions in
    `particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py`.
  - Tests: The shipped work now asserts notebook existence, exact Dynamics
    index notebook-link presence, raw-command removal, and singular feature-doc
    cross-linking in the existing example test module.

- [x] **E3-F6-P3:** Validate example execution, notebook sync, and CPU-only guidance
  - Issue: #1265 | Size: XS | Status: Shipped as the final validation pass for the published docs surface
  - Depends on: E3-F6-P1 and E3-F6-P2 producing the final runnable example and
    paired notebook artifact.
  - Goal: Re-validate the shipped example/notebook/docs surface, preserve
    CPU-only and no-temperature-feedback wording, and keep any fix limited to
    minimal wording or link alignment.
  - Files:
    `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py`,
    `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb`,
    `docs/Examples/Dynamics/index.md`,
    `docs/Features/condensation_strategy_system.md`, and
    `particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py`.
  - Tests: Re-run
    `particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py`
    as the primary docs-surface regression harness; execute notebook sync/run
    commands only to keep the published pair aligned.
