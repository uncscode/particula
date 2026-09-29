# Testing Requirements

## Coverage and co-located testing policy

1. Test coverage thresholds must NEVER be lowered.
2. Each phase must include self-contained tests.
3. Tests are committed in the same PR as the implementation.
4. Test files use `*_test.py` suffix in module-level `tests/` directories.
5. Use repository-configured full-package coverage and its normal threshold
   through `.opencode/tools/run_pytest.py` without a target.

The generic maintenance template's fixed 80% example and attribution to
`pyproject.toml` are not a new local gate. Active repository policy and E9
govern: do not pass coverage-source/threshold overrides or raw comprehensive
`--cov` controls. Focused fix checks use coverage-disabled assertion evidence.
Focused-target coverage is invalid evidence, not a fix failure; rerun focused
assertions with `--no-cov`, then the untargeted runner for coverage. Do not add
local `-Werror`; retain the configured warning policy.

## Scientific and ownership matrix

| Area | Required assertions | Phase |
|---|---|---|
| Concentration interpretation | Resolved counts at V=.25/1/4 m³, PMF sums and radius-PDF trapezoids at fixed V=1; inverse storage behavior if added; no double normalization | P1 specifies/P3 implements |
| Derived quantities | Per-particle mass distinct from population mass density and extensive inventory; radii, fractions and zero/empty cases as applicable | P1/P3 |
| Species order | Distinct sentinel values per lane; interleaved true/false flags; all-false mask; no loss or silent sorting | P1/P4 |
| Invalid alignment | Wrong box/species counts, mapping/configuration mismatch and approved duplicate/missing-name policy; no input mutation | P4 |
| Mutation and ownership | Held versus copied arrays, supplied identities per helper contract, builder kind/volume/grid rejection, explicit read-only validation after direct construction/mutation, rejection snapshots, direct mutation changes derived output | P1 specifies/P3 implements; P4 follows |
| Replacement specification | Individual mismatch, same-object replacement, accepted coordinated shape change, all-or-none rejection, no GPU rebinding | P2 specifies; M2 implements |
| Integrated helper contract | Data-only mixed-gas/non-unit-volume fixture with current-array reads, not a new runnable | P5 |
| Protected GPU boundary | Existing transfer and export tests; Warp CPU baseline when installed, optional CUDA clean skip | P4/P5 |

Use independent NumPy formulas. For resolved count storage w, calculate
c=w/V; for PMF use per-bin density directly; for PDF integrate dn/dr or
mass × dn/dr over the validated radius grid before taking a bulk result.
Assert each species' inventory
`V * (sum_i(m_i,s * c_i) + gas_concentration_s)` using the approved mapping,
not only the all-species total. Account for nonpartitioning gas separately
where it has no particle lane. Include values that differ by species so equal
shapes cannot hide bad mapping. Use explicit float64 fixtures, typically
`rtol=1e-12, atol=1e-30` for small mass inventories with a scale-based rationale;
use exact identity/unchanged-array assertions where appropriate. Do not relax
an existing tighter scientific tolerance.

Container batch dimensions remain supported as before; tests do not authorize
multi-box CPU processes. Keep process physical checks separate from builder
creation and explicit post-mutation interpretation validation. Never delete enduring
facade-fixture assertions instead of preserving/migrating their behavior.

## Implementation command sequence

Refine the focused list to the actual helper diff and add new adjacent files;
the following are future validation commands, not drafting results:

```bash
pytest particula/particles/tests/ particula/gas/tests/ -q --no-cov
pytest particula/gpu/tests/conversion_test.py particula/gpu/tests/kernel_exports_test.py -q --no-cov
pytest particula/dynamics/tests/wall_loss_strategies_test.py -q --no-cov
pytest particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py -q --no-cov
.opencode/tools/run_pytest.py
.opencode/tools/run_linters.py
mypy particula/ --ignore-missing-imports
python3 .opencode/tools/build_mkdocs.py --validate-only --strict
mkdocs build --strict
```

At P5 include the data-only integration test once its file is assigned.
Both wall-loss suites protect downstream regressions; the concrete adjacent
suite is excluded from normal recursive collection and must run explicitly.
CUDA is optional evidence, never CPU fallback; no benchmark is required.
Record command output, revision/date, counts, coverage and unavailable
dependencies. Required failing/unavailable checks block M2 admission.
