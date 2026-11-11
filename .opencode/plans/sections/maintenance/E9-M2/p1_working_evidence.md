# P1 implementation evidence

Base: `229bdb4a1`; branch: `codex/e9-m2-p1-flat-aerosol`.
Date: 2026-11-10 (America/Denver). P1 implementation is validated. PR publication
and the commit identity are recorded in the PR; this does not mark M2 complete
or admit M3. P2–P6 remain subsequent serial work.

Implemented keyword-only native construction, identity properties, structural
then representation admission, current-data string representation, and explicit
legacy aggregate routing. AerosolBuilder remains legacy until P3. The exact
transition inventory is in `legacy_consumers.md`.

Repository `.venv` validation:

| Command | Result |
|---|---|
| `pytest particula/tests/aerosol_test.py particula/tests/legacy_aerosol_test.py particula/tests/aerosol_builder_test.py particula/tests/aerosol_validation_test.py particula/integration_tests/quick_start_test.py -q --no-cov` | 72 passed |
| `.opencode/tools/run_pytest.py` with host GPU/cache access | 6,782 passed; 19 skipped; 92% coverage; configured 80% gate passed |
| `.opencode/tools/run_linters.py` | Ruff and mypy passed |
| `pytest examples_tests -q -Werror --no-cov` with host access | 150 passed; 1 skipped |
| `pytest scripts/tests -q -Werror --no-cov` | 127 passed |
| `pytest particula/dynamics/tests/wall_loss_strategies_test.py particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py -q --no-cov` with host access | 59 passed |
| `mkdocs build --strict` | Passed |
| `python .opencode/tools/build_mkdocs.py --validate-only --strict` | Passed |

The initial restricted full run failed 136 cases: it included an incorrect
quick-start native type expectation, subsequently corrected, and GPU/cache
failures. It is not passing evidence. The later host-access full run above
supersedes it. Restricted example execution also failed CUDA warning matching;
host execution passed. No GPU absence or performance claim is made.

All 21 changed notebook code-cell pairs were synchronized from their Python
sources with Jupytext and compared cell-by-cell; outputs and metadata were
preserved. The wrapper `validate_notebook.py --sync` could not import `adw`, so
Jupytext was used directly. Changed Python and notebook code cells parse.
Full tutorial workloads and regenerated outputs remain M5-owned; the source
example regression suite above was executed. Example changes route imports,
plus existing paired-source docstring/type-annotation drift from synchronization.

Local literal outputs: `/tmp/e9-m2-p1-{full,focused,lint,wall}.log`.
