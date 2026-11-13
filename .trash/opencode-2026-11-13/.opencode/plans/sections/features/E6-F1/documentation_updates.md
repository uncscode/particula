# Documentation Updates

## Completed Documentation

Issues #1389 through #1392 established and published the CPU dilution helpers,
strategy, runnable, validation, and public export boundary. `DilutionStrategy`
and `Dilution` are available from `particula.dynamics` and `par.dynamics`, while
`get_dilution_step` and `dilute_aerosol` remain concrete-module-only.

Issue #1393 completed P5 without changing those APIs. The canonical guide,
`docs/Features/dilution_strategy_system.md`, records public construction and the
same-aerosol return contract, SI units and exact exponential update, supported
CPU concentration domains, substeps, validation/no-op/rollback behavior,
protected state, helper boundary, E6/E6-F2 relationship, and explicit
non-goals. `docs/Examples/cpu_dilution.py` is a deterministic, CPU-only public
API example covering particle, partitioning-gas, and multi-species gas-only
concentrations.

The feature guide is discoverable from `docs/Features/index.md`,
`docs/Examples/index.md`, and `docs/index.md`. References to the excluded
example source use its absolute GitHub URL rather than a local MkDocs link.

`particula/tests/dilution_docs_test.py` regression-checks example execution,
result isolation, console output, import boundary, guide contract text, and
resolving local Markdown links without specialized hardware.
