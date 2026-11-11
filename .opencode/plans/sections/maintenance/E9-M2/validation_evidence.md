# E9-M2 final implementation and handoff evidence

## Scope and acceptance

All six planned phases are implemented as sequential stacked PRs. This final
phase adds developer guidance, corrected planning records, the completed
capability/temporary-consumer ledgers, and eight data-only integration cases.
P1–P5 each passed local focused, full-package and lint/type gates before the
next phase began. See their phase records for exact predecessor commits.

Final implementation revision: `f6304ff70d6ba60a2c7af2e9c01cdc409472b89c`.
All 15 final command groups completed with exit code 0. The revision remained
unchanged throughout the run. The initial evidence-only follow-up changed documentation only. Later notebook
CI repairs are recorded separately below; the package implementation remains
identical to this full-suite tested revision.

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

## Notebook CI follow-up (2026-11-10 local / 2026-11-11 UTC)

The original P1 notebook run `42949672036` was cancelled at its 15-minute
job limit while validating 21 changed notebooks serially. Earlier notebook
pair checks established source synchronization, not complete notebook execution.

P1 commits `8c6daa66a` and `77e04bef8` select the same changed-notebook set
with NUL-delimited Git output and a JSON matrix, one notebook per 15-minute
job, at most six concurrently, with fail-fast disabled and unbuffered output.
The existing 300-second notebook timeout and timeout-skip policy are unchanged.
Deleted files are excluded, filenames with spaces are preserved, and first
pushes retain the existing all-tracked-notebooks behavior. Regression tests
exercise real Git history for PR/push selection, deletion, spaces and empty diffs.

That matrix exposed two execution errors. Commit `83f6d09a4` supplies the
500-bin example's matching zero-charge array and extracts the staggered
example's one-element gas concentration with `np.asarray(...).item()`, avoiding
NumPy's rejected array-to-float conversion. Python/notebook code is synchronized
and saved outputs remain unchanged. No scientific implementation was modified.

The workflow and notebook fixes were merged sequentially through P2–P6, without
rewriting phase history. At combined revision `d940aa1b3`, local validation:

- `pytest scripts/tests -q -Werror --no-cov`: **130 passed in 26.39s**.
- The unchanged notebook validator executed both repaired notebooks:
  **2 passed, 0 failed, 0 skipped**.
- Ruff checks passed for the selector and its tests; notebook-source commit
  hooks passed.

An initial isolated-checkout infrastructure attempt picked up the later-phase
editable package in a subprocess (129 passed, 1 failed). Explicitly pointing
PYTHONPATH to P1 resolved that environment mismatch: **130 passed**. The combined
P6 run above required no such override.

Final hosted notebook validation at P1 revision
`83f6d09a47636120e04776130f79ee681288364d`:
[run 42956553611](https://github.com/uncscode/particula/actions/runs/42956553611)
completed **successfully: 20 executed passes, 0 failures, 1 timeout skip**.
The skip is `Biomass_Burning_Cloud_Interactions.ipynb`; it is not evidence of a
completed simulation. The run selected only the 21 notebooks changed in P1.
The delayed dependency-install job eventually completed and its notebook passed.
Strict documentation build passed after updating this record.
