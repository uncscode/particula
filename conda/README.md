# Conda release validation

The separate **conda-feedstock** GitHub Actions workflow builds the current
checkout with conda-build on Linux and tests it in a clean Python 3.12 conda
environment using conda-forge packages. It runs automatically only for PRs to
`main` that change the literal `__version__` value in `particula/__init__.py`.
Other edits to that file run only the inexpensive version check. Subsequent
updates to a version-changing PR rebuild its current merge checkout.

For a manual run, select **Actions → conda-feedstock → Run workflow**, then
choose the branch. GitHub exposes this control once the workflow exists on the
default branch. Manual runs do not require a version change. Normal source CI
continues to exercise GPU tests, and also tests the version gate and staging
helpers; the conda build is a separate release check.

## Local commands

With `conda-build` installed in the active conda environment:

```bash
python scripts/conda_feedstock.py
```

For a faster installed-wheel check without conda:

```bash
python scripts/run_release_tests.py
```

The wheel command creates a disposable virtual environment and installs the
built wheel, its declared dependencies, and pytest. It validates test isolation,
but does not substitute for the conda build and dependency solve.

Both routes run `pip check`, print the installed version and import location,
and execute the release suite from a temporary tests-only workspace. Application
source files are not copied into that workspace; an isolated interpreter plus
pytest's importlib mode prevents accidentally testing the source checkout.
Missing declared fixtures fail rather than silently skipping tests.

## 0.2.x release selection

The shared command is defined in `scripts/run_release_tests.py`:

```text
not slow and not performance and not benchmark
and not warp and not cuda and not gpu_parity
```

The entire `particula/gpu` test subtree is temporarily excluded before
collection, including unmarked benchmark helpers and GPU example tests. The
two GPU-only execution modules `diagnostics_test.py` and
`gpu_resources_test.py` are also excluded before their eager Warp imports.
Other execution tests and CPU integration tests remain eligible. GPU example
modules outside that subtree are Warp-marked and load example files lazily.
Resident GPU benchmark helper suites are also Warp-marked, including their
hardware-free cases.
CPU condensation, coagulation, nucleation, gas, particle, execution, and
integration domains must each contribute selected tests.

Review these exclusions for **v0.3**; do not carry them forward as an implicit
GPU validation policy. `warp-lang` remains a runtime dependency consistent with
`pyproject.toml`. Deferring GPU release validation does not make it optional.

The release test inputs are explicit:

- Test modules and their fixtures under `particula`, including integration tests.
- Root and package `conftest.py` files and `pyproject.toml`.
- `docs/Examples/cpu_dilution.py`.
- `docs/Examples/Nucleation/cpu_nucleation.py`.
- `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py`.

Documentation wording/link and planning-reference assertions were removed.
They are not a hidden release prerequisite. Documentation rendering remains
in the existing MkDocs source-checkout workflow.

## Feedstock handoff

`conda/recipe/meta.yaml` mirrors the relevant dependency/build/test contract
of conda-forge/particula-feedstock PR #53. Its source is the explicitly staged
current checkout instead of a tagged archive. Keep this mirror synchronized
when the external feedstock changes; this workflow does not fetch or execute
unreviewed remote recipe changes.

In the external feedstock, preserve its release URL/checksum and maintainer
metadata, copy the mirrored `test.source_files` list, and replace the broad
pytest invocation with:

```yaml
test:
  # Keep the source_files and requires from conda/recipe/meta.yaml.
  commands:
    - python scripts/run_release_tests.py --installed
```

The script includes `pip check` and the import/version smoke test. The package
source must contain these fixes, either through a new release or a reviewed
recipe patch to v0.2.13. Changing this branch alone does not update the existing
tag's archive. Rebuild the external feedstock after applying the handoff.

## Evidence

Conda build output and packages are retained under `.artifacts/conda-feedstock/`
and uploaded as `conda-feedstock-python312` even on failure. The log includes
the package version, installed import location, pytest results, and JSON counts
for collected, deselected, passed, skipped, failed, and errored cases. Paths
excluded before collection do not contribute to deselected counts; skipped
counts include module-level collection skips as well as skipped test cases.
The standalone wheel runner also retains `.artifacts/release-tests.log`.

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
