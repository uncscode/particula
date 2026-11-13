# M3 review fixes

The review of PR #1632 at `1bbc0f803` raised six threads covering five
independent issues. This follow-up preserves the accepted PMF/PDF coagulation
method and the M4 orchestration boundary.

| Review issue | Change and regression evidence |
|---|---|
| Density-only GPU communication accepts native raw counts (two threads) | Both direct volume evolution and particle communication reject tagged PMF/PDF/resolved populations before payload access. Legacy communication tests remain passing. |
| Finite counts can overflow N/V | Native condensation preflight checks the resolved ratio before any caller-owned writes. The `1e308 / 0.25` regression snapshots primary and output arrays. |
| Rounded condensation can overdraw gas | Evaporation budgets use representable releases. Finalized uptake is reconciled before either inventory writer; bounded downward corrections suppress uptake if needed. Independent tiny-budget regressions exercise PMF and resolved weights and direct/prepared whole calls. |
| Complete evaporation leaves occupied zero-mass particles | CPU single/staggered/latent and mapped GPU calls clear resolved counts and charge after inventory accounting. Staggered sweeps stop selecting retired particles. Independent process-sequence tests cover full removal, surviving unmapped cores, gas and latent energy, and subsequent coagulation/wall-loss admission. CPU count/charge storage ownership is validated before mutation. |
| Zero gas width becomes particle width | Native enqueue retains the validated zero width, clears transfer outputs, and skips gas-indexed work. Direct/prepared regression tests retain primary state. |

Clearing charge after complete evaporation records disappearance of the
particle; this does not add gas-phase charge transport. No PMF/PDF species
transport, communication migration, or new orchestration API is introduced.

## Validation

Use the repository `.venv`. `WARP_CACHE_PATH=/tmp/particula-m3-review-warp`
keeps compilation artifacts outside the tracked source. The focused hardware
run used both Warp CPU and CUDA; it did not substitute one for the other.

- Focused native condensation/coagulation/wall loss plus population boundaries
  and direct/parity communication: **947 passed**, no skips.
- Untargeted package validation: **9,124 passed, 19 skipped**, 454.14 seconds;
  **92% coverage**, unchanged 80% threshold. This includes 78 additional
  regression cases relative to the preserved-method revision.
- Repository Ruff check and formatting: passed (536 files already formatted).
- Repository mypy: passed (536 source files).
- Strict documentation build: passed.

Focused reproduction:

```bash
.venv/bin/pytest \
  particula/gpu/kernels/tests/condensation_native_test.py \
  particula/gpu/kernels/tests/coagulation_native_test.py \
  particula/gpu/kernels/tests/wall_loss_native_test.py \
  particula/gpu/tests/population_boundary_test.py \
  particula/gpu/kernels/tests/communication_test.py \
  particula/gpu/tests/communication_parity_test.py -q --no-cov
.venv/bin/python .opencode/tools/run_pytest.py
.venv/bin/mypy particula/ --ignore-missing-imports
.venv/bin/ruff check particula/
.venv/bin/ruff format --check particula/
.venv/bin/mkdocs build --strict
```

Local logs are retained in `.artifacts/e9-m3/review-fixes/` and are not
committed artifacts. Earlier phase and preserved-method counts describe their
respective revisions; this follow-up adds regression coverage.
