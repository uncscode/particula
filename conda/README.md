# Conda release validation

The separate **conda-feedstock** GitHub Actions workflow builds the current
checkout with conda-build on Linux and tests it in a clean Python 3.12 conda
environment using conda-forge packages. It runs automatically for PRs to
`main` that change the literal `__version__` value in `particula/__init__.py`
or release infrastructure: `conda/`, release runners and their tests, this
workflow, pytest configuration/bootstrap, or the three declared CPU examples
and their tests.
Other initializer-only edits run just the inexpensive change gate. The gate
uses the PR merge base, so subsequent updates to a qualifying PR rebuild its
current merge checkout without requiring another version bump.

For a manual run, select **Actions → conda-feedstock → Run workflow**, then
choose the branch and `feedstock_ref` (`main` by default, or for example
`refs/pull/54/head` to check a proposed feedstock handoff). GitHub exposes this
control once the workflow exists on the
default branch. Manual runs do not require a version change. Normal source CI
continues to exercise package and all source example tests, including GPU
coverage, and separately tests infrastructure under `scripts/tests`; the conda
build is a separate release check. Default pytest `testpaths` is `particula`.
Run the other source suites explicitly:

```bash
pytest examples_tests -q -Werror
pytest scripts/tests -q -Werror
```

All 13 source example test modules live in `examples_tests/`, with their
existing basenames. The marker-policy regression lives in
`scripts/tests/pytest_marker_policy_test.py`, not in the package suite.

## External feedstock drift gate

The independent `feedstock-contract` job fetches
`conda-forge/particula-feedstock` and compares its literal `test.requires`,
`test.source_files`, and ordered `test.commands` against our locally tested
recipe. Automatic PR runs compare against external `main` in **advisory drift
mode**; manual dispatch selects a candidate PR ref and enforces **strict
parity**. It reads recipe data only and executes no code,
Jinja, build scripts, or test commands from that checkout. It saves both recipe
files, their SHA-256 hashes, the external commit, and a JSON comparison report.
Unsupported test templates/selectors/schema and missing data fail explicitly
in both modes. Validation is bounded to the test contract, not the entire
recipe. Advisory mode accepts only differences between two readable,
supported test contracts and emits a visible warning. The JSON report keeps
`passed=false` for drift; `mode` identifies the policy and `check_passed`
records whether that policy passed. A green advisory job is not parity evidence.
Input/requirement list ordering is ignored; command ordering is significant.
The shared `python {{ python_min }}` requirement is compared literally; this
check does not certify external variant values or dependency solves.

**Source PR readiness** requires the real **conda-build** and source tests.
The automatic **feedstock-contract** check may pass with reported drift because
the new archive and external recipe cannot land before the source release.
This avoids making the source PR depend on its own unreleased archive. Do not
use `continue-on-error` to mask comparison or checkout failures.

**External release readiness** additionally requires a passing **strict**
contract check and a successful external feedstock build. The two jobs remain
independent. Use manual dispatch against the candidate `feedstock_ref`, then
rerun against `main` after the external recipe merges. Configuring the
automatic jobs as required source-PR checks does not replace this strict
post-handoff release validation.

This catches the PR #54 regression: the external recipe copied only
`particula/` and ran broad pytest, while our mirror supplied the required
fixtures and used the isolated CPU release runner. The previous PR check
validated only that mirror and could not observe the mismatch. The frozen
failed-recipe fixture under `scripts/tests/fixtures/` now verifies that both
the missing inputs and the wrong command fail strict validation and remain
visible in advisory PR reports.

For a local data-only comparison (requires PyYAML):

```bash
python scripts/check_feedstock_contract.py \
  --external-recipe /path/to/particula-feedstock/recipe/meta.yaml
# Pre-release comparison only: warn on supported contract differences.
python scripts/check_feedstock_contract.py \
  --external-recipe /path/to/particula-feedstock/recipe/meta.yaml --allow-drift
pytest scripts/tests -q
```

The external conda-forge rerun remains required: this bounded contract check
does not execute the external build or prove full recipe/environment parity.

## Local commands

With `conda-build` installed in the active conda environment:

```bash
python scripts/conda_feedstock.py
```

For installed-wheel checks without conda, run both suites:

```bash
python scripts/run_release_tests.py
python scripts/run_release_tests.py --suite examples
```

Each wheel command creates a disposable virtual environment and installs the
built wheel, its declared dependencies, and pytest. These validate test isolation,
but do not substitute for the conda build and dependency solve.

Both routes run `pip check`, print the installed version and import location,
and execute each release suite from its own temporary workspace. Package
application sources are not copied into those workspaces; an isolated interpreter
plus pytest's importlib mode prevents accidentally testing the source checkout.
Missing declared fixtures fail rather than silently skipping tests.

| Suite | Selection and isolated inputs |
| --- | --- |
| `--suite package` (default) | Tests, fixtures, conftests, and config; no documentation. |
| `--suite examples` | Three CPU test/script pairs plus root conftest/config. |

Conda tests the already-installed package with both commands in this order:

```bash
python scripts/run_release_tests.py --installed
python scripts/run_release_tests.py --installed --suite examples
```

## 0.2.x release selection

The shared command is defined in `scripts/run_release_tests.py`:

```text
not slow and not performance and not benchmark
and not warp and not cuda and not gpu_parity
```

The entire `particula/gpu` test subtree is temporarily excluded before
collection, including unmarked benchmark helpers. The
two GPU-only execution modules `diagnostics_test.py` and
`gpu_resources_test.py` are also excluded before their eager Warp imports.
Other execution tests and CPU integration tests remain eligible. GPU example
modules now live in `examples_tests/` and are not staged by either release suite.
Resident GPU benchmark helper suites are also Warp-marked, including their
hardware-free cases.
CPU condensation, coagulation, nucleation, gas, particle, execution, and
integration domains must each contribute selected tests.

Review these exclusions for **v0.3**; do not carry them forward as an implicit
GPU validation policy. `warp-lang` remains a runtime dependency consistent with
`pyproject.toml`. Deferring GPU release validation does not make it optional.

The package release workspace contains test modules and fixtures under
`particula`, including integration tests, root and package `conftest.py` files,
and `pyproject.toml`. It contains **zero documentation inputs** and no
`examples_tests/` or `scripts/tests/` suite.

`CPU_EXAMPLE_TESTS` in `scripts/run_release_tests.py` is the authoritative
release example-test allowlist; `CPU_EXAMPLES` declares the matching scripts.
The examples workspace contains only these three pairs plus root
`conftest.py` and `pyproject.toml`:

| Test under `examples_tests/` | Script under `docs/Examples/` |
| --- | --- |
| `dilution_example_test.py` | `cpu_dilution.py` |
| `nucleation_example_test.py` | `Nucleation/cpu_nucleation.py` |
| `condensation_latent_heat_example_test.py` | `Dynamics/Condensation/Condensation_Latent_Heat.py` |

Each declared example test module must contribute selected tests. The conda
recipe lists inputs for both suites, but the runner stages them separately;
it never copies the full documentation tree into either workspace.

Documentation wording/link, notebook publication-state, and planning-reference
assertions were intentionally removed at the maintainer's request, superseding
the original issue's proposal to retain them in source CI. Documentation
rendering remains in the existing MkDocs source-checkout workflow. Executable
CPU example checks retain numerical/runtime and public API-boundary coverage;
they require only the explicitly listed Python examples, not publication prose,
notebooks, or planning records.

## Feedstock handoff

`conda/recipe/meta.yaml` is the local release mirror, derived from the earlier
conda-forge/particula-feedstock handoff and now using the two-suite contract.
Its source is the explicitly staged current checkout instead of a tagged
archive. Keep local and external contracts synchronized for the chosen release;
the current mirror does not establish that the external recipe has changed.
The drift job fetches the external recipe for data-only comparison; the build
job executes the local reviewed mirror only.

### Immediate external fix for the unpatched v0.2.14 archive

The existing v0.2.14 archive uses the earlier layout and runner. Preserve its
release URL/checksum and maintainer metadata, and use the **seven-input,
one-command** contract from that tag rather than this checkout's new paths:

```yaml
test:
  requires:
    - python {{ python_min }}
    - pytest
    - pip
  source_files:
    - particula
    - pyproject.toml
    - conftest.py
    - scripts/run_release_tests.py
    - docs/Examples/cpu_dilution.py
    - docs/Examples/Nucleation/cpu_nucleation.py
    - docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py
  commands:
    - python scripts/run_release_tests.py --installed
```

Do not add `examples_tests/` paths or `--suite examples` to an unpatched
v0.2.14 recipe: that archive does not provide the new layout and suite option.
The immediate fix must be checked against the tagged contract, not treated as
a match for this checkout's two-suite mirror.

`conda/feedstock-pr54.patch` supplies this exact test-only change against PR
#54 revision `0a37f3ac68e52671fc9a352f3d2828918a010785`. In the feedstock
checkout on the PR branch, apply it with:

```bash
git apply --check /path/to/particula/conda/feedstock-pr54.patch
git apply /path/to/particula/conda/feedstock-pr54.patch
```

The patch preserves the release URL, checksum, dependencies, and maintainer
metadata. It is a prepared handoff; publishing it and rerunning the external
build remain required.

### New two-suite contract

Adopting the current mirror requires a **new source archive or a reviewed
recipe patch** that supplies the runner changes and relocated example tests.
Version **0.2.15** is the prepared release for this layout. After merging,
publish its GitHub release and use that archive's URL/checksum in the external
feedstock; a version commit or merge alone does not publish the release.
Only then copy the current `conda/recipe/meta.yaml` test inputs and use:

```yaml
test:
  # Keep the source_files and requires from conda/recipe/meta.yaml.
  commands:
    - python scripts/run_release_tests.py --installed
    - python scripts/run_release_tests.py --installed --suite examples
```

Each suite includes `pip check` and the import/version smoke test. Changing
this branch alone does not update an existing tag's archive or the external
feedstock. Retain maintainer metadata and use the correct source URL/checksum
for the chosen archive. Rebuild the external feedstock after applying the
handoff; neither an external recipe update nor a passing external build is
claimed here.

## Evidence

Conda build output and packages are retained under `.artifacts/conda-feedstock/`
and uploaded as `conda-feedstock-python312` even on failure. The log includes
the package version, installed import location, pytest results, and JSON counts
for collected, deselected, passed, skipped, failed, and errored cases. Paths
excluded before collection do not contribute to deselected counts; skipped
counts include module-level collection skips as well as skipped test cases.
The standalone runner retains `.artifacts/release-tests.log` for the package
suite and `.artifacts/release-examples.log` for the examples suite.

### Two-suite cleanup — 2026-10-04

On baseline `58da90e07f970d5ad28a3dc92f7bae1b92b52dde` plus the uncommitted
cleanup, both isolated Python 3.12.12 wheel suites passed for version 0.2.14:

- `python scripts/run_release_tests.py`: **3,544 collected, 768 deselected,
  2,761 passed, 18 skipped**, no failures/errors. Three skips were during
  collection. The test workspace contained no documentation tree.
- `python scripts/run_release_tests.py --suite examples`: **20 passed**, no
  skips/deselections/failures/errors, with only the declared CPU inputs staged.
- Both reported installed `site-packages` provenance and passed `pip check`.
  Dependencies were NumPy 2.5.3, SciPy 1.18.1, and Warp 1.17.0; pytest was 9.1.1.
- Full source examples: **150 passed, 1 skipped**. Infrastructure tests:
  **103 passed**. Both ran with `-Werror`.
- The prepared PR #54 patch applied cleanly to the frozen recipe and matched
  the tagged contract while preserving release metadata.
- The real conda command remains unavailable locally (`conda` is not
  installed). External publication/build/rerun and workflow dispatch remain
  pending. These successful checks are wheel/source evidence only.

### Prepared 0.2.15 release

After opening [upstream PR #1617](https://github.com/uncscode/particula/pull/1617),
both isolated wheel commands were rerun with the package bumped to **0.2.15**.
The package suite again passed **2,761 tests**, with 18 skipped and 768
deselected; all **20 CPU example tests** passed. Installed version/provenance
and `pip check` passed in both disposable environments. The suite logs now
hold these 0.2.15 results. Conda CI and external feedstock evidence remain
separate pending gates; no release tag was created by this validation.

The older results below predate the suite split.

### Local verification — 2026-10-03

- Isolated Python 3.12.12 wheel: **3,614 collected, 811 deselected,
  2,788 passed, 18 skipped, zero failures/errors**. Three skips occurred during
  collection; the remaining 15 were selected cases. `pip check` passed and
  the import/version smoke reported installed `particula 0.2.13` from the
  disposable environment's `site-packages`.
- This wheel run used NumPy 2.5.3, SciPy 1.18.1, pytest 9.1.1, and Warp 1.17.0.
  These are pip-resolved versions, not conda dependency-solve evidence.
- The untargeted repository coverage runner passed **6,786 tests**, with
  **20 skips and 93% coverage**, before the final marker-only refinements and
  splitting mixed import checks. The final affected-module/runner regression
  run passed **265 tests with one skip**. Ruff checking passed.
- A real conda build was attempted but unavailable because `conda` is not
  installed locally. The new GitHub Actions build and external feedstock rerun
  remain required; no conda CI success is claimed by the wheel result.

### PR review follow-up — 0.2.14

The package version is now `0.2.14`, enabling the PR's conda build gate.
After restoring CPU example API-boundary checks and updating the published
test commands, the isolated wheel run passed **2,791 tests**, with **18 skipped,
811 deselected, and 3,617 collected**. Installed-version smoke and `pip check`
passed. Ruff checking and strict MkDocs validation also passed. This remains
wheel evidence; the triggered conda workflow supplies separate build evidence.
