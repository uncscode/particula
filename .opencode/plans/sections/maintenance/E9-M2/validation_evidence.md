# E9-M2 final implementation and handoff evidence

## Scope and acceptance

All six planned phases are implemented as sequential stacked PRs. This final
phase adds developer guidance, corrected planning records, the completed
capability/temporary-consumer ledgers, and eight data-only integration cases.
P1–P5 each passed local focused, full-package and lint/type gates before the
next phase began. See their phase records for exact predecessor commits.

Final implementation revision: `f6304ff70d6ba60a2c7af2e9c01cdc409472b89c`.
All 15 final command groups completed with exit code 0. The revision remained
unchanged throughout the run. Subsequent evidence updates change documentation
only; the executable and test tree is identical to this tested revision.

**Final M2 acceptance is pending Kyle Gorkowski.** Metadata remains In Progress
for review and final sign-off; this does not mean construction implementation is
missing. No PR merge, release, M3 admission or later-track migration is claimed.

## Phase PRs

| Phase | PR | Implementation commit |
|---|---|---|
| P1 | https://github.com/uncscode/particula/pull/1625 | d5ec8ffb2 |
| P2 | https://github.com/uncscode/particula/pull/1626 | 19ed66933 |
| P3 | https://github.com/uncscode/particula/pull/1627 | 378aaa2c0 |
| P4 | https://github.com/uncscode/particula/pull/1628 | e24cb7990 |
| P5 | https://github.com/uncscode/particula/pull/1629 | 5e042230c |
| P6 | https://github.com/uncscode/particula/pull/1630 | f6304ff70 |

## Protected boundaries

M1's structural and representation validators are reused without redesign.
Binned distributions remain density-based at V=1; only resolved density inputs
multiply by physical volume. Aggregate construction/replacement retains supplied
identities; initializers allocate owned arrays. Species mapping stays process
owned, and unnamed ordering cannot be certified by shape checks.

Legacy scientific/runnable/example consumers retain explicit temporary imports.
The exact inventory and M6 removal conditions are in `legacy_consumers.md`.
Tagged particle upload remains rejected because the Warp metadata carrier is
absent; untagged transfer, device ownership and resident identity remain
unchanged. No GPU normalization or checkpoint support is claimed.

## Validation policy

Use repository `.venv` with host GPU/cache access for GPU-bearing tests. Record
actual available devices and skips; no skip is a measured CUDA result. Focused
checks use `--no-cov`; only the untargeted repository runner supplies coverage
with its configured threshold. Explicit wall-loss suites, source examples,
infrastructure/helper packaging, lint/type and strict docs remain required.

Notebook import-routing changes are synchronized from Python with Jupytext,
with original outputs preserved. Full tutorial workloads and refreshed outputs
remain M5-owned; executed source example suites are recorded separately.

## Final committed-revision validation

Run on 2026-11-10 using repository `.venv`: Python 3.12.12, NumPy 2.3.4,
Warp 1.11.1; CPU and native CUDA (RTX 5060, 8 GiB, sm120) available.
Host GPU/cache access was used. Local raw logs and command manifest are in
`/tmp/e9-m2-final/` (temporary local evidence, not a published artifact).

| Gate | Literal outcome |
|---|---|
| Focused native/legacy construction and M1 contracts | 258 passed |
| Retained process guards, explicitly both wall-loss suites | 243 passed |
| Transfer/export contracts | 150 passed |
| Untargeted repository coverage runner | 6888 passed, 19 skipped; 92% coverage; VALIDATION: PASSED |
| Source examples, `-Werror` | 150 passed, 1 skipped |
| Infrastructure/plugin tests, `-Werror` | 127 passed |
| Repository lint runner | Exit 0 |
| Explicit mypy | Success: no issues found in 510 source files |
| Strict documentation validation and build | Both exit 0 |
| Generated plugin synchronization | Exit 0 |
| Notebook pairs | Verified synchronized code, parsing and preserved outputs for 24 notebook pairs. |
| Developer guidance | Both Python snippets executed successfully |
| Diff whitespace check | Exit 0 |

The default coverage threshold remains 80%; no collection filter, warning-policy
relaxation, or threshold override was applied to the full runner. Existing mypy
untyped-body notes and informational documentation link messages did not fail
the configured checks. Skips remain skips, not CUDA success claims.

### Exact command ledger

Commands below ran sequentially with `.venv/bin` first on PATH. Each exited 0.
The two `/tmp` checks compare notebook code with Jupytext and original outputs,
and execute the two Python blocks in the construction guide, respectively.

- **environment**: `/home/kyle/Code/particula/.venv/bin/python -c 'import sys,warp,numpy; print(sys.version); print("NumPy",numpy.__version__); print("Warp",warp.__version__); print("Devices",warp.get_devices())'`
- **focused**: `/home/kyle/Code/particula/.venv/bin/pytest particula/tests/aerosol_test.py particula/tests/legacy_aerosol_test.py particula/tests/aerosol_replacement_test.py particula/tests/aerosol_builder_test.py particula/tests/legacy_aerosol_builder_test.py particula/tests/aerosol_validation_test.py particula/particles/tests/particle_data_initialization_test.py particula/particles/tests/resolved_initialization_test.py particula/particles/tests/particle_data_builder_test.py particula/particles/tests/representation_builders_test.py particula/particles/tests/particle_population_test.py particula/gas/tests/gas_data_builder_test.py particula/integration_tests/data_container_contract_test.py particula/integration_tests/native_aerosol_construction_test.py -q --no-cov`
- **legacy-process-guards**: `/home/kyle/Code/particula/.venv/bin/pytest particula/dynamics/tests/dilution_test.py particula/dynamics/tests/nucleation_runnable_test.py particula/dynamics/tests/wall_loss_strategies_test.py particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py -q --no-cov`
- **transfer-exports**: `/home/kyle/Code/particula/.venv/bin/pytest particula/gpu/tests/conversion_test.py particula/gpu/tests/kernel_exports_test.py -q --no-cov`
- **full**: `/home/kyle/Code/particula/.opencode/tools/run_pytest.py`
- **examples**: `/home/kyle/Code/particula/.venv/bin/pytest examples_tests -q -Werror`
- **infrastructure**: `/home/kyle/Code/particula/.venv/bin/pytest scripts/tests -q -Werror`
- **lint**: `/home/kyle/Code/particula/.opencode/tools/run_linters.py`
- **mypy**: `/home/kyle/Code/particula/.venv/bin/mypy particula/ --ignore-missing-imports`
- **docs-validation**: `/home/kyle/Code/particula/.venv/bin/python .opencode/tools/build_mkdocs.py --validate-only --strict`
- **docs**: `/home/kyle/Code/particula/.venv/bin/mkdocs build --strict`
- **plugin**: `/home/kyle/Code/particula/.venv/bin/python scripts/build_particula_plugins.py --check`
- **notebooks**: `/home/kyle/Code/particula/.venv/bin/python /tmp/e9_notebook_check.py`
- **guide**: `/home/kyle/Code/particula/.venv/bin/python /tmp/e9_guide_check.py`
- **diff-check**: `git diff --check`
