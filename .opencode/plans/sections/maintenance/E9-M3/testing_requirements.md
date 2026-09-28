# Testing Requirements

## Policy

1. Test coverage thresholds must NEVER be lowered.
2. Each phase includes self-contained co-located tests committed in the same
   PR as modified functions. Use `*_test.py` in module-level `tests/`.
3. Focused fixes use coverage-disabled assertion evidence (`--no-cov`). Folder,
   file, node, name and marker selections are not coverage evidence.
4. Coverage comes only from the full applicable suite through untargeted
   `.opencode/tools/run_pytest.py`, using repository-configured full-package
   coverage and the normal threshold. Focused-target coverage is invalid
   evidence, not a fix failure. No arbitrary gates or local source/threshold
   overrides. The generic template's numerical gate does not supersede the
   active testing guide/runner policy.
5. Do not add local `-Werror`; rely on configured warning policy and retain
   intentional warning assertions. This draft claims no test execution.

## Scientific matrix

- Native construction, process configuration, units and M1 species alignment;
  mixed/all-false partitioning and skipped species as applicable.
- V=0.25/1/4 m^3, unequal weights and multiple species. Distinguish particle
  mass, weighted density/extensive inventory and PDF integration measures.
- Condensation uptake/evaporation, isothermal/staggered/latent-heat behavior,
  current-gas coupling, limiting, zero time and inactive state.
- Supported coagulation mechanisms/combinations, binned/resolved stepping,
  per-species weighted conservation and supported charge bookkeeping.
- Wall-loss geometry/distribution matrix, neutral/charged limits, removed
  inventory and remaining-state invariants; it is a sink, not constant mass.
- Independent analytical/NumPy oracles and explicit tolerances; distinguish
  deterministic agreement, tight conservation and stochastic aggregate bounds.
- n_boxes!=1 rejection before writes, directly mutated/malformed inputs,
  configuration mismatch, return identity and documented failure semantics.

## Focused checks followed by comprehensive evidence

Run from the implementation worktree; retain literal outputs and new test
paths from the implementation diff. Broad focused directories include owning
strategy/builder/helper regressions, but still supply assertions only.

```bash
pytest particula/dynamics/condensation/ -q --no-cov
pytest particula/dynamics/coagulation/ -q --no-cov
pytest particula/dynamics/tests/wall_loss_strategies_test.py -q --no-cov
pytest particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py -q --no-cov
pytest particula/integration_tests/ -q --no-cov
pytest particula/gpu/tests/conversion_test.py particula/gpu/tests/kernel_exports_test.py -q --no-cov
.opencode/tools/run_pytest.py
.opencode/tools/run_linters.py
mypy particula/ --ignore-missing-imports
mkdocs build --strict
```

BOTH wall-loss commands are mandatory. `pyproject.toml` excludes
`particula/dynamics/wall_loss/tests/` from recursive discovery; neither the
normal suite nor untargeted coverage substitutes for direct adjacent-suite
assertions. Separate invocations also avoid same-basename collection issues.

If staggered behavior changes, retain the existing manual slow/performance
reproduction without introducing a new performance claim:

```bash
pytest particula/dynamics/condensation/tests/staggered_performance_test.py -v -m "slow and performance" --no-cov
```

E9 appendix D1–D2 requires CPU/direct/prepared GPU normalization regressions,
metadata rejection and V=0.25/1/4 conservation for each changed process family.
Warp CPU is the installed-Warp baseline for applicable tests; CUDA is optional
pass-or-clean-skip only. Record absent
runtime/device evidence. Required failures/unavailable checks block handoff;
optional skips never substitute for CPU scientific evidence.
