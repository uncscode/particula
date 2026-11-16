# Stage 1 delivery

Worktree: `/tmp/particula-stage-01`.
Branch: `codex/stage-01-scientific-correctness`.
Upstream base: `894f60350` (`uncscode/particula`, `main`).

## Completion target and evidence

All five work packages are included. The selected baseline is closed-box
monodisperse growth with dilution, complemented by constant-reservoir growth,
exact dilution, constant-kernel coagulation and stochastic removal references.
Durable results, input regimes, limits and validation commands belong to the
[scientific correctness record](../docs/contribute/validation/numerics/scientific-correctness.md).

| Work package | Delivery |
| --- | --- |
| Reproduce/classify findings | Density and negative coagulation births reproduced; phase, timestep, composition and sampling limits classified; existing tutorial warning verified. |
| Independent references | Radius-space growth quadrature, exact continuous coagulation solution, exact dilution, mixture geometry, existing activity/Kelvin limits and species-accounting tests. |
| Confirmed corrections | Additive-volume density, nonnegative interpolated collision births, import-safe logging, release-event source validation and stale version guidance. |
| Numerical uncertainty | Separate timestep/process-order, radius-grid, represented-slot and ensemble-count checks; explicit accuracy bounds and species budgets. |
| Reproducibility | Installed-wheel suite, isolated documentation bundle checks, explicit Warp dependency policy, collection audit, refreshed outputs and publication-gate review. |

Full source validation: 10583 passed, 20 skipped, one expected failure;
coverage 91.34%. Installed selection: 5086 passed, 25 skipped, 1130 deselected.
Examples: 241 passed, one skipped. Tooling: 319 passed. Ruff, formatting,
Mypy, documentation build and rendered structure checks passed.

## Delivery

Implemented, reviewed, committed and pushed in `dbd7a29d6`.
[PR #1658](https://github.com/uncscode/particula/pull/1658) targets upstream
`main` and is open and mergeable. Hosted checks were still running at delivery;
local evidence above does not claim those checks have finished. Merge and
publication remain separate actions.

Final documentation artifact rebuild/check passed: version 0.3.1 wheel
4,234,936 bytes; source archive 3,821,804 bytes. Scientific source changes were
reviewed against the independent regressions without weakening tolerances.

CPU and available CUDA evidence is scoped to the recorded tests. Experimental
validation, adaptive solvers, new phase physics and arbitrary custom-kernel
qualification remain owned by later work. This is the full bounded Stage 1
reference delivery, not an assertion that every physical model is validated.
