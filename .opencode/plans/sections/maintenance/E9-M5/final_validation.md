# M5 final validation — 2026-11-11

Base: M4 merge `3393e47fa7a880e4d9934723e93216e5b5b8425a` (still upstream main
at final check). Executable revision: `d725c7d68`; subsequent edits are
contributor/agent guidance and plan/evidence records. All commands below ran
with repository `.venv` on Python 3.12.12, NumPy 2.3.4, pytest 8.4.2 and Warp
1.11.1. Warp CPU was available; CUDA was unavailable. Warp cache was explicitly
`/tmp/particula-m5-warp`. No coverage policy or scientific tolerance changed.

| Command (repository root; `.venv/bin` on PATH) | Terminal outcome |
| --- | --- |
| `python .opencode/tools/run_pytest.py --timeout 1200` (untargeted) | PASS: 8060 passed, 678 skipped, 287.63s; assertions PASSED; coverage 92%, normal 80.0% threshold PASSED |
| `pytest examples_tests -q -Werror --no-cov` | PASS: 223 passed, 2 skipped, 50.17s |
| `pytest scripts/tests -q -Werror --no-cov` | PASS: 130 passed, 29.81s |
| `pytest particula/dynamics/tests/wall_loss_strategies_test.py particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py -q --no-cov -rs` | PASS: 59 passed, 0.07s; includes the directory omitted from normal recursive collection |
| `python .opencode/tools/run_linters.py` | PASS: Ruff check/fix/format and mypy; no production-source changes |
| `mypy particula/ --ignore-missing-imports` | PASS: no issues found in 551 source files; untyped-body informational notes retained |
| `ruff check docs/Examples examples_tests/native_theory_guides_test.py` | PASS |
| `ruff format --check docs/Examples examples_tests/native_theory_guides_test.py` | PASS: 60 files already formatted |
| `python scripts/build_particula_plugins.py --check` | PASS: six skills and host packages validated |
| `mkdocs build --strict --site-dir /tmp/m5-final-site` | PASS: 1.45s; no link notices |
| `git diff --check` | PASS |

Focused final native-access/process/theory/quick-start/raw-rate-autodiff run:
51 passed, 1 optional CUDA skip, 12.97s. Focused native-condensation and
CPU/GPU exhaustion assertions: 252 passed, 136 CUDA/opt-in benchmark skips,
1.02s. These do not replace the untargeted coverage result.

All 58 original sources and 46 notebook pairs retain their phase execution
records in `implementation.md`; the new native access walkthrough adds the
59th source. A final AST/source audit found no forbidden legacy constructor
calls in supported sources and no retained error outputs in those notebooks.
The 31 final installation-prose pair updates preserved identical code-cell ASTs
and retained previously executed outputs. Full workloads and plot inspection
were completed in the owning phases, not replaced by reduced test fixtures.

## Additional publication audit — known failure

`mkdocs build --config-file mkdocs-full.yml --strict` exited **1** after
rendering API and saved notebook pages, with **1751 Griffe warnings** in **110
unchanged production source files**. There were no non-Griffe MkDocs warning
lines. Native `Aerosol`, `AerosolBuilder`, `ParticleData`, `GasData` and
`EnvironmentData` API targets import and their rendered HTML exists. This is
not a passing full-publication strict gate. The required base strict build
above is clean; it intentionally does not generate API pages or render
notebooks. Production docstring cleanup remains follow-up, not silently
suppressed warnings or expanded M5 scope.

The full-suite summary skips are recorded as reported; they are not all
reclassified as CUDA skips. No GPU timing/capture evidence is claimed.

Detailed session logs: `/tmp/m5-final-pytest.log`, `m5-final-examples.log`,
`m5-final-scripts.log`, `m5-final-wallloss.log`, `m5-final-linters.log`,
`m5-final-mypy.log`, `m5-final-plugin.log`, `m5-final-mkdocs.log` and
`m5-full-publication.log` under `/tmp`. This tracked ledger is the durable
summary; temporary logs are not required to interpret the result.
