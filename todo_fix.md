# TODO: finish conda release-test cleanup

**Current status (2026-10-04): local cleanup and validation complete.** External
publication, conda builds/reruns, workflow dispatch, and required-check settings
remain blocked/pending. See the closeout record below; the original failure log
is preserved unchanged.

## Failure record

- Date: 2026-10-04; package: `particula 0.2.14`.
- [Failed conda-forge job](https://github.com/conda-forge/particula-feedstock/actions/runs/37183899557/job/111381756361?pr=54)
- [Feedstock PR #54](https://github.com/conda-forge/particula-feedstock/pull/54)
- Feedstock revision: `0a37f3ac68e52671fc9a352f3d2828918a010785`.
- User-supplied log summary: **50 failed, 5800 passed, 166 skipped,
  57 deselected, 94 errors in 351.17s**.
- The supplied failure summary shows missing example scripts/modules under
  `docs/Examples/` and missing `pyproject.toml` in the conda test workspace.
  The package was built, but conda rejected it during testing.

## Confirmed gap

The previous cleanup removed documentation prose/link checks but retained
example tests and some source-text/exact-output assertions. The new release
runner stages explicit CPU example inputs and excludes GPU release tests.
The external feedstock still copies only `particula/` and invokes broad pytest:

```text
pytest -W error -W "ignore::pytest.PytestUnknownMarkWarning" -m "not slow and not performance"
```

That command still selects GPU example tests. A `warp` marker alone does not
exclude a test. The feedstock has not adopted the documented release handoff.

## Checklist

### 1. Establish the current source baseline

- [x] Bring this branch up to date with the released fixes before editing
  implementation. This branch was created from the local checkout at
  `dcd6adc5211bfa963112eb3f8e5316fc3f798ac2`, which predates these commits:
  - [Release isolation fix, d457e8f](https://github.com/uncscode/particula/commit/d457e8f6399d71e503d3fb89c313d186a40ab0da)
  - [0.2.14 review follow-up, 58da90e](https://github.com/uncscode/particula/commit/58da90e07f970d5ad28a3dc92f7bae1b92b52dde)
- [x] Preserve this checklist and any unrelated work while updating the branch.
  Fast-forwarded to `58da90e07f970d5ad28a3dc92f7bae1b92b52dde`; the
  appended runner output below remains intact.

### 2. Repair the external feedstock test recipe

- [x] Prepare and verify `conda/feedstock-pr54.patch` against the frozen failed
  recipe and the tagged seven-input/one-command contract. This is ready for
  application to PR #54, not a published external update.
- [ ] Update PR #54's `recipe/meta.yaml`, preserving its release URL, checksum,
  and maintainer metadata.
- [ ] Copy the complete `test.source_files` list from the released mirror:
  - `particula`
  - `pyproject.toml`
  - `conftest.py`
  - `scripts/run_release_tests.py`
  - `docs/Examples/cpu_dilution.py`
  - `docs/Examples/Nucleation/cpu_nucleation.py`
  - `docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py`
- [ ] Replace the existing test commands with
  `python scripts/run_release_tests.py --installed`. The runner already
  performs `pip check` and installed import/version checks.
- [ ] Rerun the external conda build and retain its actual output. A local
  wheel test is not conda-build evidence.

References:

- [Released feedstock handoff](https://github.com/uncscode/particula/blob/v0.2.14/conda/README.md#feedstock-handoff)
- [Released mirrored recipe](https://github.com/uncscode/particula/blob/v0.2.14/conda/recipe/meta.yaml)
- [Released installed-test runner](https://github.com/uncscode/particula/blob/v0.2.14/scripts/run_release_tests.py)
- [Released conda build driver](https://github.com/uncscode/particula/blob/v0.2.14/scripts/conda_feedstock.py)

### 3. Finish the upstream test cleanup

- [x] Audit tests for imports/reads of `docs/`, documentation prose, planning
  files, source-substring/count/order assertions, and exact-output checks.
- [x] Remove brittle wording/source-layout assertions; retain meaningful
  numerical, conservation, public-API, and lifecycle behavior coverage.
- [x] Review exact-output checks individually: retain contractual behavior
  where justified, rather than requiring incidental guidance wording.
- [x] Address the confirmed source-text checks in
  `particula/tests/gpu_resident_graph_capture_docs_test.py`, including:
  `source.count("replay_captured_resident_graph(captured, 1.0)") == 2`
  and ordering checks using `source.index(...)`.
- [x] Review `particula/tests/pytest_marker_policy_test.py` so its
  `pyproject.toml` consistency check has an explicit source-CI home or
  declared fixture, rather than an implicit installed-package dependency.

Original paths from the failed-job summary (all 13 example tests now live in
`examples_tests/` with unchanged basenames; marker policy is in `scripts/tests/`):

- `particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py`
- `particula/execution/tests/gpu_resident_session_example_test.py`
- `particula/gpu/tests/data_containers_example_test.py`
- `particula/gpu/tests/gpu_coagulation_direct_example_test.py`
- `particula/gpu/tests/gpu_complete_process_sequence_example_test.py`
- `particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py`
- `particula/gpu/tests/gpu_direct_kernels_example_test.py`
- `particula/gpu/tests/gpu_direct_nucleation_example_test.py`
- `particula/tests/backend_selected_coagulation_example_test.py`
- `particula/tests/dilution_example_test.py`
- `particula/tests/gpu_resident_graph_capture_docs_test.py`
- `particula/tests/gpu_resident_multi_timestep_docs_test.py`
- `particula/tests/nucleation_example_test.py`
- `particula/tests/pytest_marker_policy_test.py`

[Released graph-capture example tests](https://github.com/uncscode/particula/blob/v0.2.14/particula/tests/gpu_resident_graph_capture_docs_test.py)

### 4. Make package and example test boundaries explicit

- [x] Move tests requiring `docs/Examples/` to an explicitly invoked example
  suite using a repository-consistent location and naming convention.
- [x] Update CI, test discovery, runner staging, and documented commands to
  run that suite with its declared inputs.
- [x] Ensure ordinary package testing works with `docs/` absent and exercises
  the installed package, without relying on checkout imports.
- [x] Keep useful example execution coverage active in source CI.
- [x] Keep the current 0.2.x GPU release exclusions explicit and preserve
  their planned v0.3 review; do not silently make Warp optional.
- [ ] Reconcile the upstream recipe mirror and external feedstock after any
  subsequent changes to the release-test contract.

### 5. Validate and close out

- [x] Run affected package tests from an isolated installed-package workspace
  without the documentation tree; verify import provenance.
- [x] Run the example suite with its required scripts present.
- [x] Run relevant lint and test-runner/staging regression checks.
- [x] Run `python scripts/run_release_tests.py` for isolated wheel evidence.
- [ ] Run `python scripts/conda_feedstock.py` in a conda-build environment,
  or use the upstream `conda-feedstock` workflow.
- [ ] Obtain a passing external feedstock rerun after the recipe update.
- [x] Record exact revisions, commands, selected test counts, outcomes, and
  remaining blockers. Do not mark the conda issue resolved from wheel-only
  evidence or a reduced suite that unintentionally drops behavior coverage.

## Status

### Scope audit — 2026-10-04

All 14 modules in the appended failed-job summary are listed above. The
reported 144 failures/errors break down as follows:

| Category | Failed | Errors | Required immediate repair |
| --- | ---: | ---: | --- |
| CPU condensation example | 4 | 6 | Supply declared condensation script |
| CPU dilution example | 6 | 0 | Supply declared dilution script |
| CPU nucleation example | 3 | 0 | Supply declared nucleation script |
| Pytest marker configuration | 1 | 0 | Supply `pyproject.toml` |
| GPU examples (10 modules) | 36 | 88 | Use the intended GPU-excluding release runner |
| Total | 50 | 94 | Apply external recipe handoff and rerun |

The immediate external repair is the complete recipe handoff, not adding only
`docs/` or only changing the marker expression. The required root bootstrap,
configuration, runner, and selected CPU inputs must travel together. This
handoff can use the existing v0.2.14 archive. It does not require changing
scientific algorithms, runtime dependencies, GPU behavior, or the version.

The broader example-suite separation and source-text cleanup in sections 3–4
are now implemented locally. Numerical/conservation/lifecycle checks and
intentional API behavior remain. The marker-policy test is configuration
validation in `scripts/tests/`; for the unchanged v0.2.14 archive, its immediate
external remedy is still the declared `pyproject.toml` input.

The released tree audit found the expected example paths in all 13
example-dependent modules. Extra source-layout checks include
`gpu_direct_nucleation_example_test.py` (synchronization call count) as well
as the graph-capture source-order checks already listed. Repository-text
mentions in ordinary docstrings are not runtime file dependencies.

### Fix our PR validation

- [x] Add a separate read-only external feedstock contract check. Compare
  `test.requires`, `test.source_files`, and ordered `test.commands` against
  the local mirror; do not execute remote recipe code.
- [x] Fail on mismatches, missing comparison inputs, and unsupported test
  templates/selectors/schema. Retain exact recipes, hashes, and a report.
- [x] Record the checked external commit. Automatic runs check external
  `main`; manual dispatch accepts `feedstock_ref`, including
  `refs/pull/54/head`, for candidate handoffs.
- [x] Trigger and enable the conda build for recipe, runner, workflow,
  release-test configuration, and selected CPU-example changes even without
  a version bump. Preserve the literal-version/merge-base gate otherwise.
- [x] Add an offline regression using the failed PR #54 recipe. It rejects
  the missing input list and broad pytest command, and accepts the completed
  handoff. Add gate-trigger alignment and staging-input coverage.
- [ ] Require both `conda-build` and `feedstock-contract` in repository branch
  protection where release checks are enforced (remote settings not changed).
- [ ] Run the new workflow against the candidate external recipe, then against
  `main` after its merge. Contract agreement is not dependency-solve or full
  external-build evidence.

Implemented locally in `.github/workflows/conda-feedstock.yml`,
`scripts/conda_feedstock.py`, and `scripts/check_feedstock_contract.py` with
regressions under `scripts/tests/`. Broader test relocation/removal and the
two-suite runner have since been implemented locally; the external recipe
update has not been published.

Local conda-build attempt: unavailable (`RuntimeError: conda is unavailable`)
when executing `scripts/conda_feedstock.py`. CI/external conda validation
remains pending; do not infer success from the contract or wheel tests.

### Earlier local validation — 2026-10-04 (before suite relocation)

- Focused `scripts/tests` pytest run: **44 passed**, including CLI rejection
  of the frozen PR #54 recipe and missing-recipe failures, corrected-handoff
  acceptance, release gate triggers, and staged input coverage.
- Untargeted repository pytest runner: **6,791 passed, 20 skipped, 93%
  coverage**, meeting its 80% gate (192.81 seconds).
- Ruff check and format-check passed for the two changed/new runner modules
  and their two test modules.
- The real conda build, external recipe update, remote workflow execution,
  and branch-protection settings remain outstanding. The automatic drift
  check is expected to reject the old external recipe until its handoff lands.

### Cleanup closeout — 2026-10-04

Source baseline: `58da90e07f970d5ad28a3dc92f7bae1b92b52dde`, plus the current
uncommitted cleanup and preserved pre-existing infrastructure changes. No new
commit or release tag was created. GitHub's PR API still reports external
PR #54 head `0a37f3ac68e52671fc9a352f3d2828918a010785`, branch
`regro-cf-autotick-bot:0.2.14_ha55db9`, open and unmerged.

Implemented:

- All 13 example modules moved to `examples_tests/`; source CI explicitly
  runs package, example, and infrastructure suites. Ruff CI includes the moved
  tests and scripts. Source CI now installs PyYAML for infrastructure tests.
- `--suite package` (default) stages package tests/fixtures/conftests/config
  with **zero documentation inputs**. `--suite examples` stages exactly the
  three CPU test/script pairs named by `CPU_EXAMPLE_TESTS`/`CPU_EXAMPLES`.
  Missing inputs and missing scientific/example selections fail explicitly.
  Both modes verify installed import/version provenance and run `pip check`.
- The local conda mirror, staging, and release-change gate now support both
  suites. The external drift job retains its lightweight pytest/PyYAML inputs;
  application-dependent marker probes run in ordinary source CI instead.
- Graph-capture and nucleation source-count/order assertions became observed
  lifecycle/transfer/synchronization checks. Example stdout snapshots became
  runtime state or sentinel-forwarding checks. Marker-policy source searches
  became actual collection probes. Removed the moving-bin filename assertion
  and the facade warning's incidental migration-document path requirement.
- Intentional API/asymptotic/no-readback source guards remain in GPU
  coagulation, dilution, communication, and wall-loss tests; these protect
  execution contracts rather than publication layout. No additional runtime
  documentation/planning-file dependency was found in package tests.
- The data-only feedstock checker now also rejects enclosing Jinja control
  flow/comments, including directives closing after the following YAML
  section. It checks new example inputs and the second command explicitly.

Final validation evidence:

| Command | Outcome |
| --- | --- |
| `python scripts/run_release_tests.py` | 3,544 collected, 768 deselected, 2,776 selected; **2,761 passed, 18 skipped, 0 failed/errors** (3 skips during collection), 37.79 s pytest time |
| `python scripts/run_release_tests.py --suite examples` | **20 collected/selected/passed**, no skips/deselections/failures/errors, 0.60 s |
| `python -m pytest examples_tests -q -Werror` | **150 passed, 1 skipped**, 36.13 s; skipped opt-in native-CUDA graph smoke |
| `python -m pytest scripts/tests -q -Werror` | **103 passed**, 8.15 s |
| Untargeted repository coverage runner (tester subagent) | **6,633 passed, 19 skipped; 92% coverage**, above the 80% gate |
| Ruff check and format-check: `scripts`, `examples_tests`, two changed particle test files | Passed |
| `mkdocs build --strict` (temporary build via validation wrapper) | Passed |
| `git apply --check` then `git apply` of `conda/feedstock-pr54.patch` in a disposable frozen-recipe workspace | Passed; parsed contract equals the mirror at `58da90e`, release/build/dependency/maintainer metadata preserved |
| `python scripts/conda_feedstock.py` | **Unavailable**: `RuntimeError: conda is unavailable; install conda-build in a conda environment or dispatch the conda-feedstock GitHub Actions workflow.` |

The script-execution wrapper ran the real wheel CLI. Its temporary examples
launcher forwarded `--suite examples` to the same entry point without changing
the runner. Retained literal outputs: `.artifacts/release-tests.log`,
`.artifacts/release-examples.log`, `.artifacts/examples_tests.log`, and
`.artifacts/scripts-tests.log`. Other wrapper/subagent output remains in the
session transcript. Temporary validation launchers were removed afterward.

Both wheel runs used Python 3.12.12, pytest 9.1.1, NumPy 2.5.3, SciPy 1.18.1,
and Warp 1.17.0; `pip check` passed. Final package import was
`/tmp/particula-release-wheel-llpwhn6a/venv/lib/python3.12/site-packages/particula/__init__.py`;
examples imported from the separate `particula-release-wheel-l075ledt` venv.
Both reported version 0.2.14. Final package wheel SHA-256:
`c54e4cc8575306bd3e27c629c7b2c9cc26f1dbd5d56d39ee19fcb3f563c2767d`.
Examples wheel SHA-256:
`878bdafc90d45af68f7ff805ce8d496e1c6484d45f6ffbab3fce055cb58d56d3`.
The package wheel was rebuilt after the final test-only assertion cleanup;
neither run changed scientific implementation or runtime dependencies.

Compared with the earlier 3,617-collected release record, 43 formerly
deselected GPU example cases left the package tree; 20 CPU example cases moved
to their independently validated release suite; 9 original marker-policy cases
moved to source infrastructure (now expanded to 18); and one incidental
filename assertion was removed. Thus the package collection reduction is
accounted for without removing numerical/conservation behavior.

Outstanding external work:

1. Publish `conda/feedstock-pr54.patch` to the external PR branch, preserving
   its archive metadata, and obtain the real external conda rerun. This harness
   has read-only URL access and no cross-repository file-write/workflow-dispatch
   operation; no external push, run, or settings change is claimed.
2. Run the local two-suite conda build in a conda-build environment (or CI).
   The prepared legacy patch targets the **unpatched v0.2.14 archive**. Adopting
   the new local two-suite mirror requires a new archive or reviewed source
   patch; do not add its paths/options to the old archive recipe.
3. Reconcile the external recipe with that new contract, run the drift workflow
   against its candidate ref and then `main`, and configure required checks
   where release protection is enforced. The local new-contract drift gate
   intentionally rejects the legacy external recipe, even after the immediate
   v0.2.14 repair. Contract agreement and wheels are not conda-build evidence.

## Original runner output (preserved)

linux_64_	UNKNOWN STEP	﻿2026-10-04T06:48:52.9609034Z Current runner version: '2.337.0'
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9632310Z ##[group]Runner Image Provisioner
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9633122Z Hosted Compute Agent
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9633861Z Version: 20260901.588
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9634478Z Commit: f88ec8081b781fac6c440065ac7ff9e710ce3d0b
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9635182Z Build Date: 2026-09-01T19:56:44Z
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9635855Z Worker ID: {965811b4-15f0-4afe-b4b8-b8ffedeca4da}
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9636900Z Azure Region: northcentralus
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9637543Z ##[endgroup]
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9639255Z ##[group]Operating System
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9639947Z Ubuntu
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9640507Z 24.04.5
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9641001Z LTS
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9641495Z ##[endgroup]
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9642018Z ##[group]Runner Image
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9642561Z Image: ubuntu-24.04
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9643158Z Version: 20260927.320.1
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9644338Z Included Software: https://github.com/actions/runner-images/blob/ubuntu24/20260927.320/images/ubuntu/Ubuntu2404-Readme.md
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9645801Z Image Release: https://github.com/actions/runner-images/releases/tag/ubuntu24%2F20260927.320
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9646939Z ##[endgroup]
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9649673Z ##[group]GITHUB_TOKEN Permissions
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9652471Z Actions: read
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9653059Z ArtifactMetadata: read
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9653629Z Attestations: read
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9654215Z Checks: read
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9654742Z CodeQuality: read
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9655247Z Contents: read
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9655787Z Deployments: read
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9656483Z Discussions: read
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9657060Z Drives: read
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9657562Z Issues: read
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9658279Z Metadata: read
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9658852Z Models: read
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9659447Z Packages: read
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9660072Z Pages: read
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9660605Z PullRequests: read
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9661153Z RepositoryProjects: read
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9662037Z SecurityEvents: read
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9662594Z Statuses: read
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9663150Z VulnerabilityAlerts: read
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9663784Z ##[endgroup]
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9666026Z Secret source: None
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9667105Z Cache mode: write
linux_64_	UNKNOWN STEP	2026-10-04T06:48:52.9668215Z Prepare workflow directory
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.0210664Z Prepare all required actions
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.0258410Z Getting action download info
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.2435009Z Download action repository 'actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1' (SHA:3d3c42e5aac5ba805825da76410c181273ba90b1)
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.4536001Z Complete job name: linux_64_
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.5220270Z ##[group]Run actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.5221309Z with:
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.5221859Z   repository: conda-forge/particula-feedstock
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.5225540Z   token: ***
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.5226054Z   ssh-strict: true
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.5226723Z   ssh-user: git
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.5227274Z   persist-credentials: true
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.5227837Z   clean: true
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.5228344Z   sparse-checkout-cone-mode: true
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.5228952Z   fetch-depth: 1
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.5229434Z   fetch-tags: false
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.5229959Z   show-progress: true
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.5230493Z   lfs: false
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.5230963Z   submodules: false
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.5231504Z   set-safe-directory: true
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.5232059Z   allow-unsafe-pr-checkout: false
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.5232796Z ##[endgroup]
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6184087Z Syncing repository: conda-forge/particula-feedstock
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6186091Z ##[group]Getting Git version info
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6187439Z Working directory is '/home/runner/work/particula-feedstock/particula-feedstock'
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6188706Z [command]/usr/bin/git version
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6252833Z git version 2.55.0
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6268775Z ##[endgroup]
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6277898Z Temporarily overriding HOME='/home/runner/work/_temp/66ce5832-ccfb-4613-a210-36cc5ba9e56e' before making global git config changes
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6279506Z Adding repository directory to the temporary git global config as a safe directory
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6284610Z [command]/usr/bin/git config --global --add safe.directory /home/runner/work/particula-feedstock/particula-feedstock
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6329177Z Deleting the contents of '/home/runner/work/particula-feedstock/particula-feedstock'
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6332302Z ##[group]Determining repository object format
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6334206Z ##[endgroup]
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6335707Z ##[group]Initializing the repository
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6338607Z [command]/usr/bin/git init /home/runner/work/particula-feedstock/particula-feedstock
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6439938Z hint: Using 'master' as the name for the initial branch. This default branch name
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6441665Z hint: will change to "main" in Git 3.0. To configure the initial branch name
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6443063Z hint: to use in all of your new repositories, which will suppress this warning,
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6444130Z hint: call:
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6444811Z hint:
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6445611Z hint: 	git config --global init.defaultBranch <name>
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6446741Z hint:
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6447423Z hint: Names commonly chosen instead of 'master' are 'main', 'trunk' and
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6448400Z hint: 'development'. The just-created branch can be renamed via this command:
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6449196Z hint:
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6449662Z hint: 	git branch -m <name>
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6450265Z hint:
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6451294Z hint: Disable this message with "git config set advice.defaultBranchName false"
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6452501Z Initialized empty Git repository in /home/runner/work/particula-feedstock/particula-feedstock/.git/
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6454400Z [command]/usr/bin/git remote add origin https://github.com/conda-forge/particula-feedstock
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6498595Z ##[endgroup]
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6499484Z ##[group]Disabling automatic garbage collection
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6501665Z [command]/usr/bin/git config --local gc.auto 0
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6528090Z ##[endgroup]
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6528969Z ##[group]Setting up auth
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6529830Z Removing SSH command configuration
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6533149Z [command]/usr/bin/git config --local --name-only --get-regexp core\.sshCommand
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6558798Z [command]/usr/bin/git submodule foreach --recursive sh -c "git config --local --name-only --get-regexp 'core\.sshCommand' && git config --local --unset-all 'core.sshCommand' || :"
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6862983Z Removing HTTP extra header
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6867791Z [command]/usr/bin/git config --local --name-only --get-regexp http\.https\:\/\/github\.com\/\.extraheader
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.6892513Z [command]/usr/bin/git submodule foreach --recursive sh -c "git config --local --name-only --get-regexp 'http\.https\:\/\/github\.com\/\.extraheader' && git config --local --unset-all 'http.https://github.com/.extraheader' || :"
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.7069791Z Removing includeIf entries pointing to credentials config files
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.7074654Z [command]/usr/bin/git config --local --name-only --get-regexp ^includeIf\.gitdir:
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.7100109Z [command]/usr/bin/git submodule foreach --recursive git config --local --show-origin --name-only --get-regexp remote.origin.url
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.7280181Z [command]/usr/bin/git config --file /home/runner/work/_temp/git-credentials-09aca1b2-0811-4c00-aca8-57a0ad25ac05.config http.https://github.com/.extraheader AUTHORIZATION: basic ***
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.7311127Z [command]/usr/bin/git config --local includeIf.gitdir:/home/runner/work/particula-feedstock/particula-feedstock/.git.path /home/runner/work/_temp/git-credentials-09aca1b2-0811-4c00-aca8-57a0ad25ac05.config
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.7337852Z [command]/usr/bin/git config --local includeIf.gitdir:/home/runner/work/particula-feedstock/particula-feedstock/.git/worktrees/*.path /home/runner/work/_temp/git-credentials-09aca1b2-0811-4c00-aca8-57a0ad25ac05.config
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.7368211Z [command]/usr/bin/git config --local includeIf.gitdir:/github/workspace/.git.path /github/runner_temp/git-credentials-09aca1b2-0811-4c00-aca8-57a0ad25ac05.config
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.7395786Z [command]/usr/bin/git config --local includeIf.gitdir:/github/workspace/.git/worktrees/*.path /github/runner_temp/git-credentials-09aca1b2-0811-4c00-aca8-57a0ad25ac05.config
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.7421103Z ##[endgroup]
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.7422036Z ##[group]Fetching the repository
linux_64_	UNKNOWN STEP	2026-10-04T06:48:53.7426555Z [command]/usr/bin/git -c protocol.version=2 fetch --no-tags --prune --no-recurse-submodules --depth=1 origin +6a261a2433a30adb87c2a63adc84befc43e1846b:refs/remotes/pull/54/merge
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0144014Z From https://github.com/conda-forge/particula-feedstock
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0147360Z  * [new ref]         6a261a2433a30adb87c2a63adc84befc43e1846b -> pull/54/merge
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0153137Z ##[endgroup]
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0155112Z ##[group]Determining the checkout info
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0157565Z ##[endgroup]
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0158806Z [command]/usr/bin/git sparse-checkout disable
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0207207Z [command]/usr/bin/git config --local --unset-all extensions.worktreeConfig
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0236015Z ##[group]Checking out the ref
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0239371Z [command]/usr/bin/git checkout --progress --force refs/remotes/pull/54/merge
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0289573Z Note: switching to 'refs/remotes/pull/54/merge'.
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0290363Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0290839Z You are in 'detached HEAD' state. You can look around, make experimental
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0292034Z changes and commit them, and you can discard any commits you make in this
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0293113Z state without impacting any branches by switching back to a branch.
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0293748Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0294234Z If you want to create a new branch to retain commits you create, you may
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0295401Z do so (now or later) by using -c with the switch command. Example:
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0296386Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0296791Z   git switch -c <new-branch-name>
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0297378Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0297737Z Or undo this operation with:
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0298298Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0298531Z   git switch -
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0298868Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0299453Z Turn off this advice by setting config variable advice.detachedHead to false
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0300312Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0301062Z HEAD is now at 6a261a2 Merge 0a37f3ac68e52671fc9a352f3d2828918a010785 into 1483cdb74558336330fe70cd66def1ac072abffe
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0303411Z ##[endgroup]
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0331638Z [command]/usr/bin/git log -1 --format=%H
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0352292Z 6a261a2433a30adb87c2a63adc84befc43e1846b
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0592266Z ##[group]Run if [[ "$(uname -m)" == "x86_64" ]]; then
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0593206Z ^[[36;1mif [[ "$(uname -m)" == "x86_64" ]]; then^[[0m
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0594311Z ^[[36;1m  docker run --rm --privileged multiarch/qemu-user-static:register --reset --credential yes^[[0m
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0595399Z ^[[36;1mfi^[[0m
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0995158Z shell: /usr/bin/bash --noprofile --norc -e -o pipefail {0}
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.0996418Z ##[endgroup]
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.1997013Z Unable to find image 'multiarch/qemu-user-static:register' locally
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.6175360Z register: Pulling from multiarch/qemu-user-static
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.6185480Z 205dae5015e7: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.6187712Z 816739e52091: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.6188937Z 30abb83a18eb: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.6190439Z 0657daef200b: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.6192010Z 0657daef200b: Waiting
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.6999514Z 30abb83a18eb: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.7004478Z 30abb83a18eb: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.7005983Z 816739e52091: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.7007334Z 816739e52091: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.7155636Z 205dae5015e7: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.7157567Z 205dae5015e7: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.7720052Z 0657daef200b: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.7722506Z 0657daef200b: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:48:54.7902695Z 205dae5015e7: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.0757905Z 816739e52091: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.0857785Z 30abb83a18eb: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.0982569Z 0657daef200b: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.1027908Z Digest: sha256:f654e8c1dda9d99f6ec95db08c5019d8bed9a9f4170c21350ea86692712304ef
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.1045659Z Status: Downloaded newer image for multiarch/qemu-user-static:register
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.2849764Z Setting /usr/bin/qemu-alpha-static as binfmt interpreter for alpha
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.2855875Z Setting /usr/bin/qemu-arm-static as binfmt interpreter for arm
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.2864569Z Setting /usr/bin/qemu-armeb-static as binfmt interpreter for armeb
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.2871828Z Setting /usr/bin/qemu-sparc-static as binfmt interpreter for sparc
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.2879564Z Setting /usr/bin/qemu-sparc32plus-static as binfmt interpreter for sparc32plus
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.2887190Z Setting /usr/bin/qemu-sparc64-static as binfmt interpreter for sparc64
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.2895588Z Setting /usr/bin/qemu-ppc-static as binfmt interpreter for ppc
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.2903331Z Setting /usr/bin/qemu-ppc64-static as binfmt interpreter for ppc64
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.2910900Z Setting /usr/bin/qemu-ppc64le-static as binfmt interpreter for ppc64le
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.2918272Z Setting /usr/bin/qemu-m68k-static as binfmt interpreter for m68k
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.2926333Z Setting /usr/bin/qemu-mips-static as binfmt interpreter for mips
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.2933891Z Setting /usr/bin/qemu-mipsel-static as binfmt interpreter for mipsel
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.2942918Z Setting /usr/bin/qemu-mipsn32-static as binfmt interpreter for mipsn32
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.2950034Z Setting /usr/bin/qemu-mipsn32el-static as binfmt interpreter for mipsn32el
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.2957970Z Setting /usr/bin/qemu-mips64-static as binfmt interpreter for mips64
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.2965221Z Setting /usr/bin/qemu-mips64el-static as binfmt interpreter for mips64el
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.2973065Z Setting /usr/bin/qemu-sh4-static as binfmt interpreter for sh4
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.2980352Z Setting /usr/bin/qemu-sh4eb-static as binfmt interpreter for sh4eb
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.2988061Z Setting /usr/bin/qemu-s390x-static as binfmt interpreter for s390x
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.2995629Z Setting /usr/bin/qemu-aarch64-static as binfmt interpreter for aarch64
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3003360Z Setting /usr/bin/qemu-aarch64_be-static as binfmt interpreter for aarch64_be
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3009980Z Setting /usr/bin/qemu-hppa-static as binfmt interpreter for hppa
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3018112Z Setting /usr/bin/qemu-riscv32-static as binfmt interpreter for riscv32
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3025353Z Setting /usr/bin/qemu-riscv64-static as binfmt interpreter for riscv64
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3033056Z Setting /usr/bin/qemu-xtensa-static as binfmt interpreter for xtensa
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3041092Z Setting /usr/bin/qemu-xtensaeb-static as binfmt interpreter for xtensaeb
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3048847Z Setting /usr/bin/qemu-microblaze-static as binfmt interpreter for microblaze
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3056982Z Setting /usr/bin/qemu-microblazeel-static as binfmt interpreter for microblazeel
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3064357Z Setting /usr/bin/qemu-or1k-static as binfmt interpreter for or1k
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3078183Z Setting /usr/bin/qemu-hexagon-static as binfmt interpreter for hexagon
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3890684Z ##[group]Run export CONDA_BLD_PATH="${CONDA_BLD_PATH/#~/${HOME}}"
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3891215Z ^[[36;1mexport CONDA_BLD_PATH="${CONDA_BLD_PATH/#~/${HOME}}"^[[0m
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3891662Z ^[[36;1mexport FEEDSTOCK_NAME="$(basename $GITHUB_REPOSITORY)"^[[0m
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3892125Z ^[[36;1mexport GIT_BRANCH="${GITHUB_REF_NAME}"^[[0m
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3892562Z ^[[36;1mexport MINIFORGE_HOME="${MINIFORGE_HOME/#~/${HOME}}"^[[0m
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3893035Z ^[[36;1mexport flow_run_id="github_$GITHUB_RUN_ID"^[[0m
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3893530Z ^[[36;1mexport remote_url="https://github.com/$GITHUB_REPOSITORY"^[[0m
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3894072Z ^[[36;1mexport sha="$GITHUB_SHA"^[[0m
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3894515Z ^[[36;1mif [[ "${GITHUB_EVENT_NAME}" == "pull_request" ]]; then^[[0m
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3894913Z ^[[36;1m  export IS_PR_BUILD="True"^[[0m
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3895236Z ^[[36;1melse^[[0m
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3895542Z ^[[36;1m  export IS_PR_BUILD="False"^[[0m
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3895865Z ^[[36;1mfi^[[0m
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3896443Z ^[[36;1m./.scripts/run_docker_build.sh^[[0m
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3939763Z shell: /usr/bin/bash --noprofile --norc -e -o pipefail {0}
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3940210Z env:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3940503Z   CI: github_actions
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3940871Z   CONDA_BLD_PATH: build_artifacts
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3941193Z   CONDA_FORGE_DOCKER_RUN_ARGS: 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3941543Z   CONFIG: linux_64_
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3941862Z   MINIFORGE_HOME: ~/miniforge3
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3942250Z   DOCKER_IMAGE: quay.io/condaforge/linux-anvil-x86_64:alma10
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3942660Z   UPLOAD_PACKAGES: true
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3942987Z   RATTLER_BUILD_COLOR: always
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3943359Z   RATTLER_BUILD_ENABLE_GITHUB_INTEGRATION: true
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3943733Z   BINSTAR_TOKEN: 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3944061Z   FEEDSTOCK_TOKEN: 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3944363Z   STAGING_BINSTAR_TOKEN: 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.3944667Z ##[endgroup]
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.4044367Z ##[group]Configure Docker
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.4046919Z + DOCKER_EXECUTABLE=docker
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.4051998Z +++ dirname ./.scripts/run_docker_build.sh
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.4060178Z ++ cd ./.scripts
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.4060796Z ++ pwd
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.4061675Z + THISDIR=/home/runner/work/particula-feedstock/particula-feedstock/.scripts
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.4064695Z ++ basename /home/runner/work/particula-feedstock/particula-feedstock/.scripts
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.4072227Z + PROVIDER_DIR=.scripts
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.4077182Z +++ dirname ./.scripts/run_docker_build.sh
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.4085153Z ++ cd ./.scripts/..
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.4085588Z ++ pwd
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.4090265Z + FEEDSTOCK_ROOT=/home/runner/work/particula-feedstock/particula-feedstock
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.4091166Z + RECIPE_ROOT=/home/runner/work/particula-feedstock/particula-feedstock/recipe
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.4091961Z + '[' -z particula-feedstock ']'
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.4092523Z + [[ 6a261a2433a30adb87c2a63adc84befc43e1846b == '' ]]
linux_64_	UNKNOWN STEP	2026-10-04T06:49:01.4093069Z + docker info
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8360156Z Client: Docker Engine - Community
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8360910Z  Version:    28.0.4
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8361307Z  Context:    default
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8361597Z  Debug Mode: false
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8362078Z  Plugins:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8362474Z   buildx: Docker Buildx (Docker Inc.)
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8363014Z     Version:  v0.37.1
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8363553Z     Path:     /usr/libexec/docker/cli-plugins/docker-buildx
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8364074Z   compose: Docker Compose (Docker Inc.)
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8364508Z     Version:  v2.38.2
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8364923Z     Path:     /usr/libexec/docker/cli-plugins/docker-compose
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8365323Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8365442Z Server:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8365758Z  Containers: 0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8366272Z   Running: 0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8366642Z   Paused: 0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8367039Z   Stopped: 0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8367470Z  Images: 7
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8367839Z  Server Version: 28.0.4
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8368669Z  Storage Driver: overlay2
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8369165Z   Backing Filesystem: extfs
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8369590Z   Supports d_type: true
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8370024Z   Using metacopy: false
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8370449Z   Native Overlay Diff: false
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8370859Z   userxattr: false
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8371331Z  Logging Driver: json-file
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8371795Z  Cgroup Driver: systemd
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8372247Z  Cgroup Version: 2
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8372611Z  Plugins:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8372935Z   Volume: local
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8373428Z   Network: bridge host ipvlan macvlan null overlay
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8374319Z   Log: awslogs fluentd gcplogs gelf journald json-file local splunk syslog
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8374985Z  Swarm: inactive
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8375390Z  Runtimes: io.containerd.runc.v2 runc
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8376014Z  Default Runtime: runc
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8376922Z  Init Binary: docker-init
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8399216Z ++ id -u
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8399772Z  containerd version: ee2735368117d2eb259779949d5e75cdafec9761
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8400267Z  runc version: v1.5.1-0-g8f2685a4
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8400612Z  init version: de40ad0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8400964Z  Security Options:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8401253Z   apparmor
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8401502Z   seccomp
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8401830Z    Profile: builtin
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8402117Z   cgroupns
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8402432Z  Kernel Version: 6.17.0-1022-azure
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8402774Z  Operating System: Ubuntu 24.04.5 LTS
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8403093Z  OSType: linux
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8403405Z  Architecture: x86_64
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8403713Z  CPUs: 4
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8404011Z  Total Memory: 15.61GiB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8404324Z  Name: runnervm8df0l
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8404633Z  ID: 96bc9d58-7015-4b4a-8357-8e592f0b19b5
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8405039Z  Docker Root Dir: /var/lib/docker
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8405363Z  Debug Mode: false
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8405688Z  Username: githubactions
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8406001Z  Experimental: false
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8406462Z  Insecure Registries:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8410897Z   ::1/128
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8411190Z   127.0.0.0/8
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8411506Z  Live Restore Enabled: false
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8411792Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8415175Z + export HOST_USER_ID=1001
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8415649Z + HOST_USER_ID=1001
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8416077Z + hash docker-machine
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8417161Z + ARTIFACTS=/home/runner/work/particula-feedstock/particula-feedstock/build_artifacts
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8420757Z + '[' -z linux_64_ ']'
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8421206Z + '[' -z quay.io/condaforge/linux-anvil-x86_64:alma10 ']'
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8421769Z + mkdir -p /home/runner/work/particula-feedstock/particula-feedstock/build_artifacts
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8431321Z + DONE_CANARY=/home/runner/work/particula-feedstock/particula-feedstock/build_artifacts/conda-forge-build-done-linux_64_
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8432983Z + rm -f /home/runner/work/particula-feedstock/particula-feedstock/build_artifacts/conda-forge-build-done-linux_64_
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8440229Z + DOCKER_RUN_ARGS=
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8440737Z + '[' -z github_actions ']'
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8441175Z + VOLUME_SUFFIX=,z
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8442961Z + '[' docker = podman ']'
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8443653Z + [[ -n /home/runner/work/_temp/_runner_file_commands/step_summary_6c7dac71-a6cf-4ab4-bb72-b309f3b51a2c ]]
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8444769Z + DOCKER_RUN_ARGS=' -v /home/runner/work/_temp/_runner_file_commands/step_summary_6c7dac71-a6cf-4ab4-bb72-b309f3b51a2c:/home/conda/github_step_summary:rw,z,delegated -e GITHUB_STEP_SUMMARY=/home/conda/github_step_summary'
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8446838Z ##[endgroup]
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8450335Z ##[group]Start Docker
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8451593Z + export UPLOAD_PACKAGES=true
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8452048Z + UPLOAD_PACKAGES=true
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8452374Z + export IS_PR_BUILD=True
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8452677Z + IS_PR_BUILD=True
linux_64_	UNKNOWN STEP	2026-10-04T06:49:03.8453291Z + docker pull quay.io/condaforge/linux-anvil-x86_64:alma10
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4264791Z alma10: Pulling from condaforge/linux-anvil-x86_64
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4267572Z 653c5d8d0d66: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4268366Z 53bc4a5333ab: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4269130Z 465161e8289d: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4269774Z a8fafe8103fa: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4270363Z 57e22217b0a3: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4270931Z 0312f36d85a1: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4271534Z dc60f6ac7e11: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4272058Z 7ecc323e96bf: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4272623Z e484072bd80e: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4273526Z 9c6175b8478a: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4274165Z a0faaa61e1ed: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4274670Z 4f4fb700ef54: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4275688Z 4411db1de63a: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4276279Z d249e963c18d: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4276918Z 7efef81fb680: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4277464Z 01d47cd5559f: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4277851Z 683e58154bf3: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4278215Z e4fb159ebe44: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4278525Z 41d4f592f20d: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4279076Z b559beb61628: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4279476Z 95f14765c7cd: Pulling fs layer
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4279825Z dc60f6ac7e11: Waiting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4280228Z 7ecc323e96bf: Waiting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4280536Z e484072bd80e: Waiting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4280821Z 9c6175b8478a: Waiting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4281232Z a0faaa61e1ed: Waiting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4281515Z 4f4fb700ef54: Waiting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4281863Z 4411db1de63a: Waiting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4282162Z d249e963c18d: Waiting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4282467Z a8fafe8103fa: Waiting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4283247Z 57e22217b0a3: Waiting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4283671Z 7efef81fb680: Waiting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4284005Z 0312f36d85a1: Waiting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4284368Z 01d47cd5559f: Waiting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4284673Z b559beb61628: Waiting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4285157Z 95f14765c7cd: Waiting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4285447Z e4fb159ebe44: Waiting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4285776Z 683e58154bf3: Waiting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.4286228Z 41d4f592f20d: Waiting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.6080658Z 53bc4a5333ab: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.6082202Z 53bc4a5333ab: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.6353520Z 465161e8289d: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.6354255Z 465161e8289d: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.7705816Z a8fafe8103fa: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.7707179Z a8fafe8103fa: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.9626666Z 0312f36d85a1: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:49:04.9627675Z 0312f36d85a1: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:05.1248317Z dc60f6ac7e11: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:49:05.1249241Z dc60f6ac7e11: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:05.2842718Z 653c5d8d0d66: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:49:05.2843940Z 653c5d8d0d66: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:05.2917813Z 7ecc323e96bf: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:05.4502592Z 9c6175b8478a: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:49:05.4509803Z 9c6175b8478a: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:05.4534328Z e484072bd80e: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:05.6014566Z a0faaa61e1ed: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:49:05.6016324Z a0faaa61e1ed: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:05.6673306Z 4f4fb700ef54: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:49:05.6673893Z 4f4fb700ef54: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:05.7304824Z 4411db1de63a: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:49:05.7305733Z 4411db1de63a: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:05.8050289Z d249e963c18d: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:49:05.8053038Z d249e963c18d: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:05.8690087Z 7efef81fb680: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:49:05.8690725Z 7efef81fb680: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:05.9360966Z 01d47cd5559f: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:49:05.9361905Z 01d47cd5559f: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:06.1154687Z e4fb159ebe44: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:49:06.1155554Z e4fb159ebe44: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:06.1693605Z 683e58154bf3: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:49:06.1694529Z 683e58154bf3: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:06.3016408Z b559beb61628: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:49:06.3017426Z b559beb61628: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:06.4314886Z 95f14765c7cd: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:49:06.4315985Z 95f14765c7cd: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:07.3550305Z 653c5d8d0d66: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:07.8793013Z 57e22217b0a3: Verifying Checksum
linux_64_	UNKNOWN STEP	2026-10-04T06:49:07.8793472Z 57e22217b0a3: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:09.5644311Z 53bc4a5333ab: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:10.7270744Z 465161e8289d: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:10.9276252Z a8fafe8103fa: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:11.0579683Z 57e22217b0a3: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:11.1715704Z 0312f36d85a1: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:11.2641537Z dc60f6ac7e11: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:11.3237979Z 7ecc323e96bf: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:11.3994524Z e484072bd80e: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:11.4665421Z 9c6175b8478a: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:11.4753974Z a0faaa61e1ed: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:11.4888301Z 4f4fb700ef54: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:11.5003506Z 4411db1de63a: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:11.5097900Z d249e963c18d: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:11.5985656Z 7efef81fb680: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:11.6380867Z 01d47cd5559f: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:13.2241672Z 683e58154bf3: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:13.4837138Z e4fb159ebe44: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:14.0152127Z 41d4f592f20d: Download complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.1306618Z 41d4f592f20d: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.1405901Z b559beb61628: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.1526414Z 95f14765c7cd: Pull complete
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.1561656Z Digest: sha256:f5004c1b61aced91b220f16d5efc4dda865f2ece69d8c69f7ff5582fabc47645
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.1577245Z Status: Downloaded newer image for quay.io/condaforge/linux-anvil-x86_64:alma10
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.1588576Z quay.io/condaforge/linux-anvil-x86_64:alma10
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.1602127Z + docker run -v /home/runner/work/_temp/_runner_file_commands/step_summary_6c7dac71-a6cf-4ab4-bb72-b309f3b51a2c:/home/conda/github_step_summary:rw,z,delegated -e GITHUB_STEP_SUMMARY=/home/conda/github_step_summary -v /home/runner/work/particula-feedstock/particula-feedstock/recipe:/home/conda/recipe_root:rw,z,delegated -v /home/runner/work/particula-feedstock/particula-feedstock:/home/conda/feedstock_root:rw,z,delegated -e BUILD_OUTPUT_ID -e BUILD_WITH_CONDA_DEBUG -e CI -e CONFIG -e CPU_COUNT -e FEEDSTOCK_NAME -e GIT_BRANCH -e GITHUB_ACTIONS -e HOST_USER_ID -e IS_PR_BUILD -e RATTLER_BUILD_COLOR -e RATTLER_BUILD_ENABLE_GITHUB_INTEGRATION -e UPLOAD_ON_BRANCH -e UPLOAD_PACKAGES -e flow_run_id -e remote_url -e sha -e BINSTAR_TOKEN -e FEEDSTOCK_TOKEN -e STAGING_BINSTAR_TOKEN quay.io/condaforge/linux-anvil-x86_64:alma10 bash /home/conda/feedstock_root/.scripts/build_steps.sh
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.2677759Z bash: cannot set terminal process group (-1): Inappropriate ioctl for device
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.2678468Z bash: no job control in this shell
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.3156643Z useradd: warning: the home directory /home/conda already exists.
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.3157411Z useradd: Not copying any file from skel directory into it.
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.6073174Z + export FEEDSTOCK_ROOT=/home/conda/feedstock_root
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.6073954Z + FEEDSTOCK_ROOT=/home/conda/feedstock_root
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.6074693Z + source /home/conda/feedstock_root/.scripts/logging_utils.sh
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.6076657Z ##[endgroup]
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.6081435Z ##[group]Configuring conda
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.6083419Z + export PYTHONUNBUFFERED=1
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.6084203Z + PYTHONUNBUFFERED=1
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.6084691Z + export RECIPE_ROOT=/home/conda/recipe_root
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.6085250Z + RECIPE_ROOT=/home/conda/recipe_root
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.6085935Z + export CI_SUPPORT=/home/conda/feedstock_root/.ci_support
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.6086717Z + CI_SUPPORT=/home/conda/feedstock_root/.ci_support
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.6087797Z + export CONFIG_FILE=/home/conda/feedstock_root/.ci_support/linux_64_.yaml
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.6088633Z + CONFIG_FILE=/home/conda/feedstock_root/.ci_support/linux_64_.yaml
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.6089634Z + export RATTLER_CACHE_DIR=/home/conda/feedstock_root/build_artifacts/pkg_cache
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.6090452Z + RATTLER_CACHE_DIR=/home/conda/feedstock_root/build_artifacts/pkg_cache
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.6091091Z + cat
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.6106367Z ++ date +%Y-%m-%d-%H-%M-%S
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.6118564Z + mv /opt/conda/conda-meta/history /opt/conda/conda-meta/history.2026-10-04-06-49-35
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.6150554Z + echo
linux_64_	UNKNOWN STEP	2026-10-04T06:49:35.6152124Z + micromamba install --root-prefix /home/conda/.conda --prefix /opt/conda --yes --override-channels --channel conda-forge --strict-channel-priority pip python=3.14 conda-build conda-forge-ci-setup=4 'conda-build>=26.3'
linux_64_	UNKNOWN STEP	2026-10-04T06:49:36.2681491Z Fetch Shard Index for conda-forge/linux-64                                                      ⧖ Starting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:36.3313145Z Fetch Shard Index for conda-forge/linux-64                                                ✔ Done (0.1 sec)
linux_64_	UNKNOWN STEP	2026-10-04T06:49:36.3314449Z Fetch Shard Index for conda-forge/noarch                                                        ⧖ Starting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:36.3804134Z Fetch Shard Index for conda-forge/noarch                                                  ✔ Done (0.0 sec)
linux_64_	UNKNOWN STEP	2026-10-04T06:49:36.3805635Z Fetching and Parsing Packages' Shards                                                           ⧖ Starting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:41.1136281Z Fetching and Parsing Packages' Shards                                                     ✔ Done (4.7 sec)
linux_64_	UNKNOWN STEP	2026-10-04T06:49:41.9400815Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:41.9401576Z Pinned packages:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:41.9401823Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:41.9402124Z   - su-exec==0.2
linux_64_	UNKNOWN STEP	2026-10-04T06:49:41.9402332Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:41.9402486Z Pinned packages:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:41.9402671Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:41.9402780Z   - tini==0.19.0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:41.9402969Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:41.9403619Z Resolving Environment                                                                           ⧖ Starting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1625424Z Resolving Environment                                                                     ✔ Done (0.2 sec)
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1628439Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1631629Z Transaction
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1631904Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1632099Z   Prefix: /opt/conda
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1632563Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1632720Z   Updating specs:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1633038Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1633194Z    - pip
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1633586Z    - python=3.14
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1633997Z    - conda-build
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1634481Z    - conda-forge-ci-setup=4
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1634980Z    - conda-build>=26.3
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1635218Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1635224Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1764727Z warning  libmamba Security Warning: This transaction includes executing package scripts (pre/post-link/unlink) if present. These scripts can contain arbitrary code. Please ensure you trust the package sources.
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1766512Z   Package                         Version  Build              Channel          Size
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1767565Z ─────────────────────────────────────────────────────────────────────────────────────
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1768151Z   Install:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1768809Z ─────────────────────────────────────────────────────────────────────────────────────
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1769239Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1769569Z   + cloudpickle                     3.1.2  pyhcf101f3_1       conda-forge      27kB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1770311Z   + conda-env                       2.6.0  1                  conda-forge       2kB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1771111Z   + conda-forge-ci-setup           4.27.2  py314hb783b04_101  conda-forge      88kB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1771926Z   + conda-forge-metadata        2026.9.10  pyh5ded981_0       conda-forge      23kB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1772734Z   + conda-oci-mirror                0.2.3  pyhd8ed1ab_1       conda-forge      34kB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1773798Z   + deprecated                      3.0.0  pyhc364b38_0       conda-forge      23kB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1774517Z   + joblib                          1.6.0  pyhcf101f3_0       conda-forge     229kB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1775195Z   + jq                              1.8.2  h280c20c_1         conda-forge     317kB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1775873Z   + libsodium                      1.0.22  hebe6cf0_3         conda-forge     271kB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1776857Z   + nvidia-virtual-packages         0.2.0  pyhcf101f3_0       conda-forge      15kB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1777633Z   + oniguruma                      6.9.10  hebe6cf0_1         conda-forge     281kB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1778452Z   + oras-py                        0.1.14  pyhd8ed1ab_0       conda-forge      34kB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1779229Z   + pygithub                       2.10.0  pyhd8ed1ab_0       conda-forge     189kB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1779930Z   + pynacl                          1.6.2  py311h2d59bce_5    conda-forge       2MB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1780704Z   + rattler-build                  0.76.1  hf01adef_0         conda-forge      20MB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1781505Z   + rattler-build-conda-compat     1.4.19  pyhd8ed1ab_0       conda-forge      53kB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1782252Z   + shyaml                          0.6.2  pyhd3deb0d_0       conda-forge      22kB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1782975Z   + wrapt                           2.5.0  py314hfe1a184_0    conda-forge     158kB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1783667Z   + zstandard                      0.25.0  py314hfe1a184_4    conda-forge     473kB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1784132Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1784360Z   Summary:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1784554Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1784756Z   Install: 19 packages
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1784988Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1785168Z   Total download: 24MB
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1785419Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1785908Z ─────────────────────────────────────────────────────────────────────────────────────
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1786620Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1786626Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1786633Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.1786810Z Transaction starting
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.4686975Z Linking nvidia-virtual-packages-0.2.0-pyhcf101f3_0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.4778042Z Linking conda-env-2.6.0-1
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.4779016Z Linking shyaml-0.6.2-pyhd3deb0d_0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.4798098Z Linking cloudpickle-3.1.2-pyhcf101f3_1
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.4803587Z Linking oras-py-0.1.14-pyhd8ed1ab_0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.4832739Z Linking joblib-1.6.0-pyhcf101f3_0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.4945417Z Linking rattler-build-0.76.1-hf01adef_0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.5185591Z Linking wrapt-2.5.0-py314hfe1a184_0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.5238665Z Linking libsodium-1.0.22-hebe6cf0_3
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.5357316Z Linking zstandard-0.25.0-py314hfe1a184_4
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.5387883Z Linking oniguruma-6.9.10-hebe6cf0_1
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.5406484Z Linking pynacl-1.6.2-py311h2d59bce_5
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.5581389Z Linking jq-1.8.2-h280c20c_1
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.5620910Z Linking deprecated-3.0.0-pyhc364b38_0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.5638749Z Linking conda-oci-mirror-0.2.3-pyhd8ed1ab_1
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.5667253Z Linking pygithub-2.10.0-pyhd8ed1ab_0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.5876283Z Linking conda-forge-metadata-2026.9.10-pyh5ded981_0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.5913830Z Linking rattler-build-conda-compat-1.4.19-pyhd8ed1ab_0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.5952373Z Linking conda-forge-ci-setup-4.27.2-py314hb783b04_101
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.8941861Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.8942628Z Transaction finished
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.8943026Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.9015859Z + export CONDA_LIBMAMBA_SOLVER_NO_CHANNELS_FROM_INSTALLED=1
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.9016865Z + CONDA_LIBMAMBA_SOLVER_NO_CHANNELS_FROM_INSTALLED=1
linux_64_	UNKNOWN STEP	2026-10-04T06:49:42.9017858Z + setup_conda_rc /home/conda/feedstock_root /home/conda/recipe_root /home/conda/feedstock_root/.ci_support/linux_64_.yaml
linux_64_	UNKNOWN STEP	2026-10-04T06:49:47.3062583Z + source run_conda_forge_build_setup
linux_64_	UNKNOWN STEP	2026-10-04T06:49:47.3063408Z ++ export PYTHONUNBUFFERED=1
linux_64_	UNKNOWN STEP	2026-10-04T06:49:47.3063918Z ++ PYTHONUNBUFFERED=1
linux_64_	UNKNOWN STEP	2026-10-04T06:49:47.3064368Z ++ conda config --env --set show_channel_urls true
linux_64_	UNKNOWN STEP	2026-10-04T06:49:47.9198960Z ++ conda config --env --set auto_update_conda false
linux_64_	UNKNOWN STEP	2026-10-04T06:49:48.5292559Z ++ conda config --env --set add_pip_as_python_dependency false
linux_64_	UNKNOWN STEP	2026-10-04T06:49:49.1473134Z ++ conda config --env --append aggressive_update_packages ca-certificates
linux_64_	UNKNOWN STEP	2026-10-04T06:49:49.7672648Z ++ conda config --env --remove-key aggressive_update_packages
linux_64_	UNKNOWN STEP	2026-10-04T06:49:50.3811401Z ++ conda config --env --append aggressive_update_packages ca-certificates
linux_64_	UNKNOWN STEP	2026-10-04T06:49:51.0010619Z ++ conda config --env --append aggressive_update_packages certifi
linux_64_	UNKNOWN STEP	2026-10-04T06:49:51.6923479Z ++ export CONDA_BLD_PATH=/home/conda/feedstock_root/build_artifacts
linux_64_	UNKNOWN STEP	2026-10-04T06:49:51.6924949Z ++ CONDA_BLD_PATH=/home/conda/feedstock_root/build_artifacts
linux_64_	UNKNOWN STEP	2026-10-04T06:49:51.6925572Z ++ set +u
linux_64_	UNKNOWN STEP	2026-10-04T06:49:51.6925862Z ++ case "$CI" in
linux_64_	UNKNOWN STEP	2026-10-04T06:49:51.6927919Z +++ cat /home/conda/feedstock_root/conda-forge.yml
linux_64_	UNKNOWN STEP	2026-10-04T06:49:51.6928663Z +++ shyaml get-value channel_priority strict
linux_64_	UNKNOWN STEP	2026-10-04T06:49:51.7269320Z ++ conda config --env --set channel_priority strict
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3465421Z ++ [[ ! -z '' ]]
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3466414Z ++ '[' '!' -z linux_64_ ']'
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3467085Z ++ '[' '!' -z github_actions ']'
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3467545Z ++ echo ''
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3468007Z ++ echo CI:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3468430Z ++ echo '- github_actions'
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3468919Z ++ echo ''
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3469304Z ++ set -u
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3469788Z ++ mkdir -p /opt/conda/etc/conda/activate.d
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3491379Z ++ echo 'export CONDA_BLD_PATH='\''/home/conda/feedstock_root/build_artifacts'\'''
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3498912Z ++ '[' -n '' ']'
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3499539Z ++ echo 'export PYTHONUNBUFFERED='\''1'\'''
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3503828Z +++ cat /home/conda/feedstock_root/.ci_support/linux_64_.yaml
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3505902Z +++ shyaml get-value cuda_compiler_version.0 None
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3840218Z ++ CUDA_VERSION=None
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3840685Z ++ [[ None != \N\o\n\e ]]
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3849854Z ++++ dirname /opt/conda/bin/run_conda_forge_build_setup
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3867049Z +++ cd /opt/conda/bin
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3867664Z +++ pwd
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3868166Z ++ SCRIPT_DIR=/opt/conda/bin
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3868673Z ++ source /opt/conda/bin/cross_compile_support.sh
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3873483Z +++++ dirname /opt/conda/bin/cross_compile_support.sh
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3885638Z ++++ cd /opt/conda/bin
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3886028Z ++++ pwd
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3888137Z +++ SCRIPT_DIR=/opt/conda/bin
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3893806Z ++++ conda info --json
linux_64_	UNKNOWN STEP	2026-10-04T06:49:52.3896765Z ++++ jq -r .platform
linux_64_	UNKNOWN STEP	2026-10-04T06:49:53.2788538Z +++ BUILD_PLATFORM=linux-64
linux_64_	UNKNOWN STEP	2026-10-04T06:49:53.2789090Z +++ '[' -f /home/conda/feedstock_root/.ci_support/linux_64_.yaml ']'
linux_64_	UNKNOWN STEP	2026-10-04T06:49:53.2794703Z ++++ cat /home/conda/feedstock_root/.ci_support/linux_64_.yaml
linux_64_	UNKNOWN STEP	2026-10-04T06:49:53.2795868Z ++++ shyaml get-value host_platform.0 None
linux_64_	UNKNOWN STEP	2026-10-04T06:49:53.3121593Z +++ HOST_PLATFORM=None
linux_64_	UNKNOWN STEP	2026-10-04T06:49:53.3127177Z ++++ cat /home/conda/feedstock_root/.ci_support/linux_64_.yaml
linux_64_	UNKNOWN STEP	2026-10-04T06:49:53.3128124Z ++++ shyaml get-value target_platform.0 None
linux_64_	UNKNOWN STEP	2026-10-04T06:49:53.3445083Z +++ TARGET_PLATFORM=None
linux_64_	UNKNOWN STEP	2026-10-04T06:49:53.3450809Z ++++ cat /home/conda/feedstock_root/.ci_support/linux_64_.yaml
linux_64_	UNKNOWN STEP	2026-10-04T06:49:53.3451886Z ++++ shyaml get-value cuda_compiler_version.0 None
linux_64_	UNKNOWN STEP	2026-10-04T06:49:53.3778951Z +++ CUDA_COMPILER_VERSION=None
linux_64_	UNKNOWN STEP	2026-10-04T06:49:53.3784389Z ++++ cat /home/conda/feedstock_root/.ci_support/linux_64_.yaml
linux_64_	UNKNOWN STEP	2026-10-04T06:49:53.3785033Z ++++ shyaml get-value microarch_level.0 1
linux_64_	UNKNOWN STEP	2026-10-04T06:49:53.4115679Z +++ MICROARCH_LEVEL_NEEDED=1
linux_64_	UNKNOWN STEP	2026-10-04T06:49:53.4116478Z +++ '[' None = None ']'
linux_64_	UNKNOWN STEP	2026-10-04T06:49:53.4116937Z +++ '[' None = None ']'
linux_64_	UNKNOWN STEP	2026-10-04T06:49:53.4117600Z +++ TARGET_PLATFORM=linux-64
linux_64_	UNKNOWN STEP	2026-10-04T06:49:53.4118105Z +++ HOST_PLATFORM=linux-64
linux_64_	UNKNOWN STEP	2026-10-04T06:49:53.4137819Z ++++ conda info --json
linux_64_	UNKNOWN STEP	2026-10-04T06:49:53.4138621Z ++++ jq -r '.virtual_pkgs[] | select(.[0] == "__glibc")[1]'
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3099536Z +++ DOCKER_GLIBC_VERSION=2.39
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3100712Z +++ GLIBC_VERSION=2.39
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3101195Z +++ CUDA_COMPILER_VERSION=None
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3101769Z +++ [[ linux-64 != \l\i\n\u\x\-\6\4 ]]
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3102559Z +++ [[ linux-64 == \l\i\n\u\x\-\6\4 ]]
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3103210Z +++ [[ 1 == \4 ]]
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3105246Z ++ '[' -f /home/conda/feedstock_root/.ci_support/linux_64_.yaml ']'
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3110464Z +++ cat /home/conda/feedstock_root/.ci_support/linux_64_.yaml
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3111388Z +++ shyaml get-value MACOSX_DEPLOYMENT_TARGET.0 0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3443349Z ++ mdt=0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3446760Z +++ cat /home/conda/feedstock_root/.ci_support/linux_64_.yaml
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3447405Z +++ shyaml get-value MACOSX_SDK_VERSION.0 0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3778590Z ++ msv=0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3779044Z ++ [[ 0 != \0 ]]
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3779374Z ++ [[ 0 != \0 ]]
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3779712Z ++ '[' '!' -z linux_64_ ']'
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3780213Z ++ cat /home/conda/feedstock_root/.ci_support/linux_64_.yaml
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3789553Z channel_sources:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3790236Z ++ conda info
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3790797Z - conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3791467Z channel_targets:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3792131Z - conda-forge main
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3792551Z docker_image:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3793049Z - quay.io/condaforge/linux-anvil-x86_64:alma10
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3793552Z python_min:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3793957Z - '3.12'
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3794184Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3794346Z CI:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3794627Z - github_actions
linux_64_	UNKNOWN STEP	2026-10-04T06:49:54.3794772Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1441606Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1442484Z      active environment : base
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1443279Z     active env location : /opt/conda
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1443687Z             shell level : 1
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1444072Z        user config file : /home/conda/.condarc
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1444586Z  populated config files : /opt/conda/.condarc
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1445117Z                           /opt/conda/condarc.d/anaconda-auth.yml
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1445583Z                           /home/conda/.condarc
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1446019Z           conda version : 26.7.3
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1446551Z     conda-build version : 26.7.1
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1446963Z          python version : 3.14.7.final.0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1447388Z                  solver : libmamba (default)
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1447852Z        virtual packages : __archspec=1=icelake
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1448217Z                           __conda=26.7.3=0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1448546Z                           __glibc=2.39=0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1448891Z                           __linux=6.17.0=0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1449232Z                           __unix=0=0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1449565Z        base environment : /opt/conda  (writable)
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1450024Z       conda av data dir : /opt/conda/etc/conda
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1450374Z   conda av metadata url : None
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1450833Z            channel URLs : https://conda.anaconda.org/conda-forge/linux-64
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1451318Z                           https://conda.anaconda.org/conda-forge/noarch
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1451803Z           package cache : /home/conda/feedstock_root/build_artifacts/pkg_cache
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1452256Z                           /opt/conda/pkgs
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1452630Z           notices cache : /home/conda/.cache/conda/notices
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1453006Z        envs directories : /opt/conda/envs
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1453400Z                           /home/conda/.conda/envs
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1453763Z     temporary directory : /tmp
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1454129Z                platform : linux-64
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1454825Z              user-agent : conda/26.7.3 requests/2.34.2 CPython/3.14.7 Linux/6.17.0-1022-azure almalinux/10.2 glibc/2.39 solver/libmamba conda-libmamba-solver/26.7.0 libmambapy/2.9.0
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1455515Z                 UID:GID : 1001:1001
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1455853Z              netrc file : None
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1456622Z            offline mode : False
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.1456854Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.2735967Z ++ conda config --env --show-sources
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7807334Z ==> /opt/conda/.condarc <==
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7808027Z add_pip_as_python_dependency: False
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7808467Z auto_update_conda: False
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7808937Z aggressive_update_packages:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7809417Z   - ca-certificates
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7809903Z   - certifi
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7810262Z channel_priority: strict
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7810614Z channels:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7810957Z   - conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7811312Z show_channel_urls: True
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7812020Z conda_build:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7812328Z   pkg_format: 2
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7812686Z   zstd_compression_level: 19
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7812936Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7813166Z ==> /opt/conda/condarc.d/anaconda-auth.yml <==
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7813580Z channel_settings:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7814431Z   - channel: https://repo.anaconda.cloud/*
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7815009Z     auth: anaconda-auth
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7815217Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7815350Z ==> /home/conda/.condarc <==
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7815710Z pkgs_dirs:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7816095Z   - /home/conda/feedstock_root/build_artifacts/pkg_cache
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7816850Z   - /opt/conda/pkgs
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7817185Z solver: libmamba
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7817495Z conda-build:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7817889Z   root-dir: /home/conda/feedstock_root/build_artifacts
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7818169Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7818303Z ==> envvars <==
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7818626Z allow_softlinks: False
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7819045Z bld_path: /home/conda/feedstock_root/build_artifacts
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.7819331Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:49:55.8850882Z ++ conda list --show-channel-urls
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7116048Z # packages in environment at /opt/conda:
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7121445Z #
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7122716Z # Name                        Version          Build                 Channel
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7124033Z _openmp_mutex                 4.5              20_gnu                conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7125469Z _python_abi3_support          1.0              hd8ed1ab_3            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7126993Z anaconda-auth                 0.15.3           pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7128222Z anaconda-cli-base             0.8.2            pyhc364b38_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7129533Z anaconda-client               1.14.1           pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7130767Z annotated-doc                 0.0.5            pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7132073Z annotated-types               0.8.0            pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7133363Z anyio                         4.15.1           pyh5ded981_1          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7134483Z archspec                      0.2.5            pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7135689Z attrs                         26.1.0           pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7136972Z backports                     1.0              pyhd8ed1ab_5          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7138200Z backports.tarfile             1.2.0            pyhcf101f3_2          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7139547Z backports.zstd                1.7.0            py314h680f03e_1       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7140759Z beautifulsoup4                4.15.0           pyha770c72_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7141997Z boltons                       26.2.0           pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7143167Z brotli-python                 1.2.0            py314hcd2bdb6_4       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7144342Z bzip2                         1.0.8            hda65f42_10           conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7145524Z c-ares                        1.34.8           hebe6cf0_2            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7146835Z ca-certificates               2026.7.22        hbd8a1cb_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7148489Z cached-property               2.0.1            pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7149713Z cached_property               2.0.1            pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7150933Z certifi                       2026.7.22        pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7152109Z cffi                          2.1.1            py314h8d76f0c_3       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7153214Z chardet                       7.6.0            pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7154481Z charset-normalizer            3.5.2            h1114479_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7155829Z click                         8.5.0            pyh5ded981_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7157163Z cloudpickle                   3.1.2            pyhcf101f3_1          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7158395Z colorama                      0.4.6            pyhd8ed1ab_1          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7159520Z conda                         26.7.3           py314h9e666f3_0       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7160738Z conda-build                   26.7.1           pyh31ec981_1          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7161863Z conda-env                     2.6.0            1                     conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7163208Z conda-forge-ci-setup          4.27.2           py314hb783b04_101     conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7164575Z conda-forge-metadata          2026.9.10        pyh5ded981_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7165821Z conda-index                   0.13.0           pyh5ded981_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7167236Z conda-libmamba-solver         26.7.0           pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7168569Z conda-lockfiles               0.2.2            pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7169891Z conda-oci-mirror              0.2.3            pyhd8ed1ab_1          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7171180Z conda-package-handling        2.6.0            pyh5ded981_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7172509Z conda-package-streaming       0.13.0           pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7173812Z conda-pypi                    0.13.0           pyh5ded981_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7175059Z conda-rattler-solver          0.2.0            pyh5ded981_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7176559Z conda-self                    0.3.0            pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7177709Z cpp-expected                  1.3.1            h171cf75_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7178889Z cpython                       3.14.7           py314hd8ed1ab_106     conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7180147Z cryptography                  50.0.1           py311hc91d8b8_1       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7181299Z dbus                          1.16.2           he8c428d_2            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7182526Z defusedxml                    0.7.1            pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7183718Z deprecated                    3.0.0            pyhc364b38_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7184850Z distro                        1.9.0            pyhd8ed1ab_1          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7186266Z evalidate                     2.0.5            pyhe01879c_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7187528Z filelock                      4.0.7            pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7188649Z fmt                           12.1.0           h76c4fd7_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7189831Z frozendict                    2.4.7            pyh851646a_2          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7190943Z git                           2.56.0           pl5440h3795e67_0      conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7192073Z h11                           0.16.0           pyhcf101f3_1          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7193188Z h2                            4.4.1            pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7194520Z hpack                         4.2.0            pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7195697Z httpcore                      1.0.9            pyh29332c3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7197007Z httpx                         0.28.1           pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7198364Z hyperframe                    6.1.0            pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7199474Z icu                           78.3             py310h44b86e0_2       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7200624Z idna                          3.20             pyh5ded981_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7202012Z importlib-metadata            9.0.1            pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7203428Z importlib_resources           7.1.0            pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7204725Z jaraco.classes                3.4.0            pyhcf101f3_3          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7205952Z jaraco.context                6.1.2            pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7207279Z jaraco.functools              4.6.0            pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7208359Z jeepney                       0.9.0            pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7209093Z jinja2                        3.1.6            pyhcf101f3_1          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7210108Z joblib                        1.6.0            pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7210901Z jq                            1.8.2            h280c20c_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7211941Z jsonschema                    4.26.0           pyhcf101f3_1          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7212892Z jsonschema-specifications     2025.9.1         pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7213721Z jupyter_core                  5.9.1            pyhc90fa1f_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7214487Z keyring                       25.7.0           pyha804496_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7215183Z keyutils                      1.6.3            h7cc23a3_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7215932Z krb5                          1.22.2           hbc21106_2            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7216716Z lcms2                         2.19.1           h9073bf1_3            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7217421Z ld_impl_linux-64              2.46.1           default_hbd61a6d_102  conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7218143Z lerc                          4.2.0            hdb68285_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7218802Z libarchive                    3.8.9            gpl_h3152399_101      conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7219535Z libcurl                       8.22.0           ha042cf0_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7220225Z libdeflate                    1.25             hd45a770_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7220999Z libedit                       3.1.20250104     pl5321h373387f_1      conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7221699Z libev                         4.33             h280c20c_3            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7222365Z libexpat                      2.8.5            hd2095e1_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7223059Z libffi                        3.7.0            h81df57d_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7223768Z libfreetype                   2.14.3           ha770c72_2            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7224450Z libfreetype6                  2.14.3           h5e6c136_2            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7225258Z libgcc                        16.2.0           ha9f2e26_7            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7225927Z libglib                       2.90.0           h569388d_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7226875Z libgomp                       16.2.0           he0feb66_7            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7227541Z libiconv                      1.18             h0cb94f2_3            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7228474Z libjpeg-turbo                 3.2.0            hb03c661_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7229205Z liblief                       1.0.0            h45ba95f_5            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7229857Z liblzma                       5.8.3            hb03c661_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7230549Z libmamba                      2.9.0            py314h6ba947b_1       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7231251Z libmamba-spdlog               2.9.0            h98e1848_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7231990Z libmambapy                    2.9.0            py314h79a9cf0_1       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7232928Z libmpdec                      4.0.0            hb03c661_2            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7233588Z libmsgpack-c                  6.1.0            h54a6638_7            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7234299Z libnghttp2                    1.68.1           h74cf4be_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7234993Z libpng                        1.6.58           h922cc85_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7235716Z libpsl                        0.23.1           hd9e3e90_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7236640Z libpython                     3.14.7           hdc7f604_106_cp314    conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7237315Z libsodium                     1.0.22           hebe6cf0_3            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7238001Z libsolv                       0.7.40           h72ddc62_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7238708Z libsqlite                     3.53.4           h13e7031_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7239382Z libssh2                       1.11.1           h6154650_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7240065Z libstdcxx                     16.2.0           h934c35e_7            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7240769Z libtiff                       4.7.2            hcc2c06a_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7241448Z libuuid                       2.42.4           hcfc3c73_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7242212Z libwebp-base                  1.6.0            hd42ef1d_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7242872Z libxcb                        1.17.0           hb83e432_2            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7243588Z libxcrypt                     4.4.38           h280c20c_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7244235Z libxml2                       2.15.4           h7df9aa5_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7244879Z libxml2-16                    2.15.4           hf3af7cc_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7245682Z libzlib                       1.3.2            h25fd6f3_3            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7246582Z lz4-c                         1.10.0           hee9eb32_2            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7247249Z lzo                           2.10             hebe6cf0_1003         conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7247904Z mamba                         2.9.0            h33f7037_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7248586Z markdown-it-py                4.2.0            pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7249369Z markupsafe                    3.0.3            py314h67df5f8_1       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7250041Z mbedtls                       4.0.0            hecca717_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7250734Z mdurl                         0.1.2            pyhd8ed1ab_1          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7251371Z menuinst                      2.5.2            pyhba3e0a1_3          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7252073Z more-itertools                11.1.0           pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7252920Z msgpack-python                1.2.3            py314h5383ef5_0       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7253650Z nbformat                      5.11.1           pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7254336Z ncurses                       6.6              hdb14827_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7255234Z nlohmann_json-abi             3.12.0           h0f90c79_2            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7256374Z nvidia-virtual-packages       0.2.0            pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7257234Z oniguruma                     6.9.10           hebe6cf0_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7257929Z openjpeg                      2.5.4            heb1ab33_2            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7258610Z openssl                       3.6.4            h781a0a9_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7259368Z oras-py                       0.1.14           pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7260032Z packaging                     26.3             pyhc364b38_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7260980Z patch                         2.8              h280c20c_1003         conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7261633Z patchelf                      0.19.1           hee9eb32_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7262310Z pcre2                         10.47            h8b3dc9c_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7262966Z perl                          5.44.0           1_h7cc23a3_perl5      conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7263634Z pillow                        12.3.0           py314h50bfbbb_4       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7264415Z pip                           26.2.1           pyh145f28c_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7265073Z pixi                          0.81.0           hf01adef_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7265749Z pkce                          1.0.3            pyhd8ed1ab_1          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7266693Z pkginfo                       1.12.1.2         pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7267469Z platformdirs                  4.12.2           pyh5ded981_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7268217Z pluggy                        1.6.0            pyhf9edf01_1          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7269070Z psutil                        7.2.2            py314hfe1a184_3       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7269839Z pthread-stubs                 0.4              h7cc23a3_1004         conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7270518Z py-lief                       1.0.0            py314h82f3a7e_5       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7271229Z py-rattler                    0.26.0           py311hc91d8b8_0       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7271950Z py-rattler-build              0.73.0           py311h30674b1_0       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7272610Z pybind11-abi                  11               hc364b38_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7273257Z pycosat                       0.6.6            py314h89acca1_5       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7273975Z pycparser                     3.0              pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7274771Z pydantic                      2.13.5           pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7275479Z pydantic-core                 2.46.5           py314h7e8cd81_2       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7276397Z pydantic-settings             2.15.0           pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7277080Z pygithub                      2.10.0           pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7277789Z pygments                      2.21.0           pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7278479Z pyjwt                         2.15.1           pyh5ded981_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7279107Z pynacl                        1.6.2            py311h2d59bce_5       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7279802Z pyproject_hooks               1.2.0            pyhd8ed1ab_1          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7280544Z pysocks                       1.7.1            pyha55dd90_7          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7281273Z python                        3.14.7           hcd007b5_106_cp314    conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7281968Z python-build                  1.6.1            pyhc364b38_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7282713Z python-dateutil               2.9.0.post0      pyhe01879c_2          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7283638Z python-dotenv                 1.2.3            pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7284402Z python-fastjsonschema         2.22.2           pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7285170Z python-gil                    3.14.7           h4df99d1_106          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7285975Z python-installer              1.0.1            pyh332efcf_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7286968Z python-libarchive-c           5.3              pyhe01879c_1          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7287680Z python_abi                    3.14             9_cp314               conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7288495Z pytz                          2026.4           pyh5ded981_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7289189Z pyyaml                        6.0.3            py314h67df5f8_1       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7289846Z rattler-build                 0.76.1           hf01adef_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7290562Z rattler-build-conda-compat    1.4.19           pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7291394Z readchar                      4.2.2            pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7292019Z readline                      8.3              hd6e31c0_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7292739Z referencing                   0.37.0           pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7293436Z reproc                        14.2.8.post0     hee9eb32_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7294113Z reproc-cpp                    14.2.8.post0     h1c70be6_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7294849Z requests                      2.34.2           pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7295529Z requests-toolbelt             1.0.0            pyhd8ed1ab_1          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7296462Z rich                          15.0.0           pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7297107Z ripgrep                       15.2.0           hf19af3b_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7297724Z rpds-py                       2026.6.3         py314h7e8cd81_2       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7298435Z ruamel.yaml                   0.19.1           pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7299105Z ruamel.yaml.clib              0.2.15           py314hfe1a184_5       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7299856Z secretstorage                 3.5.0            pyhc9edb4d_3          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7300581Z semver                        3.1.0            pyh5ded981_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7301288Z setuptools                    84.0.0           pyh332efcf_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7302006Z shellingham                   1.5.4            pyhd8ed1ab_2          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7302653Z shyaml                        0.6.2            pyhd3deb0d_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7303337Z simdjson                      4.6.11           h4dbf13b_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7303999Z six                           1.17.0           pyhe01879c_1          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7304629Z sniffio                       1.3.1            pyhd8ed1ab_2          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7305333Z soupsieve                     2.10             pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7306018Z spdlog                        1.17.0           hb6aa676_2            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7306990Z su-exec                       0.2              h7cc23a3_1004         conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7307647Z tini                          0.19.0           h280c20c_2            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7308317Z tk                            8.6.13           noxft_h1df4ec4_4      conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7308950Z tomli                         2.4.1            pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7309598Z tomli-w                       1.2.0            pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7310372Z tomlkit                       0.15.1           pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7311024Z tqdm                          4.70.1           pyhfa0c392_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7311687Z traitlets                     5.16.1           pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7312302Z truststore                    0.10.4           pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7312957Z typer                         0.27.2           pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7313552Z typing-extensions             4.16.0           h69aa097_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7314706Z typing-inspection             0.4.4            pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7315407Z typing_extensions             4.16.0           pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7316049Z tzdata                        2026c            h151e31d_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7317074Z unearth                       0.18.3           pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7317867Z urllib3                       2.8.0            pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7318708Z wrapt                         2.5.0            py314hfe1a184_0       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7319472Z xorg-libxau                   1.0.12           h7cc23a3_2            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7320170Z xorg-libxdmcp                 1.1.5            h7cc23a3_2            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7320884Z yaml                          0.2.5            hebe6cf0_3            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7321493Z yaml-cpp                      0.8.0            h54a6638_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7322200Z zipp                          4.1.0            pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7322867Z zlib-ng                       2.3.3            hce19668_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7323632Z zstandard                     0.25.0           py314hfe1a184_4       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.7324386Z zstd                          1.5.7            hb78ec9c_7            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:49:56.8685288Z + make_build_number /home/conda/feedstock_root /home/conda/recipe_root /home/conda/feedstock_root/.ci_support/linux_64_.yaml
linux_64_	UNKNOWN STEP	2026-10-04T06:49:58.1494614Z + [[ -f /home/conda/feedstock_root/LICENSE.txt ]]
linux_64_	UNKNOWN STEP	2026-10-04T06:49:58.1496820Z ##[endgroup]
linux_64_	UNKNOWN STEP	2026-10-04T06:49:58.1497491Z + cp /home/conda/feedstock_root/LICENSE.txt /home/conda/recipe_root/recipe-scripts-license.txt
linux_64_	UNKNOWN STEP	2026-10-04T06:49:58.1514041Z + [[ 0 == 1 ]]
linux_64_	UNKNOWN STEP	2026-10-04T06:49:58.1514570Z + CONDA_SUBDIR=linux-64
linux_64_	UNKNOWN STEP	2026-10-04T06:49:58.1517368Z + conda-build /home/conda/recipe_root -m /home/conda/feedstock_root/.ci_support/linux_64_.yaml --suppress-variables --clobber-file /home/conda/feedstock_root/.ci_support/clobber_linux_64_.yaml --extra-meta flow_run_id=github_37183899557 remote_url=https://github.com/conda-forge/particula-feedstock sha=6a261a2433a30adb87c2a63adc84befc43e1846b
linux_64_	UNKNOWN STEP	2026-10-04T06:49:58.6820149Z Adding in variants from internal_defaults
linux_64_	UNKNOWN STEP	2026-10-04T06:49:58.6821205Z Adding in variants from /home/conda/recipe_root/conda_build_config.yaml
linux_64_	UNKNOWN STEP	2026-10-04T06:49:58.6822198Z Adding in variants from /home/conda/feedstock_root/.ci_support/linux_64_.yaml
linux_64_	UNKNOWN STEP	2026-10-04T06:49:58.7447846Z WARNING: Number of parsed outputs does not match detected raw metadata blocks. Identified output block may be wrong! If you are using Jinja conditionals to include or exclude outputs, consider using `skip: true  # [condition]` instead.
linux_64_	UNKNOWN STEP	2026-10-04T06:49:59.0863625Z Attempting to finalize metadata for particula
linux_64_	UNKNOWN STEP	2026-10-04T06:50:57.2640345Z Reloading output folder: ...working... done
linux_64_	UNKNOWN STEP	2026-10-04T06:50:57.4519192Z Getting pinned dependencies: ...working... done
linux_64_	UNKNOWN STEP	2026-10-04T06:51:05.4395728Z Reloading output folder: ...working... done
linux_64_	UNKNOWN STEP	2026-10-04T06:51:05.6081866Z Getting pinned dependencies: ...working... done
linux_64_	UNKNOWN STEP	2026-10-04T06:51:05.6292823Z BUILD START: ['particula-0.2.14-pyhd8ed1ab_0.conda']
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.1100185Z Reloading output folder: ...working... done
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2782479Z Solving environment (_h_env): ...working... done
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2977870Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2979186Z ## Package Plan ##
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2979687Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2981349Z   environment location: /home/conda/feedstock_root/build_artifacts/particula_1791096598735/_h_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_p
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2983202Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2988102Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2988890Z The following NEW packages will be INSTALLED:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2989729Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2990022Z     _openmp_mutex:    4.5-20_gnu                  conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2990592Z     bzip2:            1.0.8-hda65f42_10           conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2991193Z     ca-certificates:  2026.7.22-hbd8a1cb_0        conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2991649Z     flit-core:        3.12.0-pyhcf101f3_2         conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2992081Z     icu:              78.3-py310h44b86e0_2        conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2992532Z     ld_impl_linux-64: 2.46.1-default_hbd61a6d_102 conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2992987Z     libexpat:         2.8.5-hd2095e1_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2993422Z     libffi:           3.7.0-h81df57d_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2993843Z     libgcc:           16.2.0-ha9f2e26_7           conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2994231Z     libgomp:          16.2.0-he0feb66_7           conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2994661Z     liblzma:          5.8.3-hb03c661_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2995093Z     libnsl:           2.0.1-hb9d3cd8_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2995518Z     libpython:        3.12.14-h0c77377_3_cpython  conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2995957Z     libsqlite:        3.53.4-h13e7031_1           conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2996689Z     libstdcxx:        16.2.0-h934c35e_7           conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2997155Z     libuuid:          2.42.4-hcfc3c73_0           conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2997625Z     libxcrypt:        4.4.38-h280c20c_0           conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2998034Z     libzlib:          1.3.2-h25fd6f3_3            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2998482Z     ncurses:          6.6-hdb14827_1              conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2998894Z     openssl:          3.6.4-h781a0a9_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2999325Z     packaging:        26.3-pyhc364b38_0           conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.2999739Z     pip:              26.2.1-pyh8b19718_0         conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.3000200Z     python:           3.12.14-h5f976f7_3_cpython  conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.3000649Z     readline:         8.3-hd6e31c0_1              conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.3001072Z     setuptools:       84.0.0-pyh332efcf_0         conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.3002041Z     tk:               8.6.13-noxft_h1df4ec4_4     conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.3002551Z     tzdata:           2026c-h151e31d_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.3002934Z     wheel:            0.48.0-pyhd8ed1ab_0         conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.3003386Z     zstd:             1.5.7-hb78ec9c_7            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:07.3003641Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:09.2707455Z Preparing transaction: ...working... done
linux_64_	UNKNOWN STEP	2026-10-04T06:51:10.2343950Z Verifying transaction: ...working... done
linux_64_	UNKNOWN STEP	2026-10-04T06:51:12.3175389Z WARNING:conda.conda_pypi.main:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:12.3176582Z   Did you know? You can install many PyPI packages with conda
linux_64_	UNKNOWN STEP	2026-10-04T06:51:12.3177063Z   using conda-pypi. Get started:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:12.3177457Z     https://bit.ly/4xXYt0B
linux_64_	UNKNOWN STEP	2026-10-04T06:51:12.3177731Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:12.3177869Z Executing transaction: ...working... done
linux_64_	UNKNOWN STEP	2026-10-04T06:51:14.9589030Z Reloading output folder: ...working... done
linux_64_	UNKNOWN STEP	2026-10-04T06:51:17.3875456Z Solving environment (_test_env): ...working... done
linux_64_	UNKNOWN STEP	2026-10-04T06:51:17.4059145Z Source cache directory is: /home/conda/feedstock_root/build_artifacts/src_cache
linux_64_	UNKNOWN STEP	2026-10-04T06:51:17.4060333Z Downloading source to cache: v0.2.14_0abf2359a9.tar.gz
linux_64_	UNKNOWN STEP	2026-10-04T06:51:17.4061248Z Downloading https://github.com/uncscode/particula/archive/refs/tags/v0.2.14.tar.gz
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.0117143Z Success
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.0194336Z Extracting download
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6374352Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6375220Z Rendered as:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6375887Z ```yaml
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6376443Z package:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6376864Z   name: particula
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6377362Z   version: 0.2.14
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6377804Z source:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6378761Z   url: https://github.com/uncscode/particula/archive/refs/tags/v0.2.14.tar.gz
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6379731Z   sha256: 0abf2359a9fa0802423dba110524eb72870379ce70e99fd241d247c85a6a9ca3
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6380391Z build:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6380764Z   number: '0'
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6381228Z   noarch: python
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6382937Z   script: /home/conda/feedstock_root/build_artifacts/particula_1791096598735/_h_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_p/bin/python
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6384700Z     -m pip install . -vv
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6385265Z requirements:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6385661Z   host:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6386235Z     - _openmp_mutex 4.5 20_gnu
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6386741Z     - bzip2 1.0.8 hda65f42_10
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6387232Z     - ca-certificates 2026.7.22 hbd8a1cb_0
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6387825Z     - flit-core 3.12.0 pyhcf101f3_2
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6388335Z     - icu 78.3 py310h44b86e0_2
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6388893Z     - ld_impl_linux-64 2.46.1 default_hbd61a6d_102
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6389462Z     - libexpat 2.8.5 hd2095e1_0
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6389937Z     - libffi 3.7.0 h81df57d_1
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6390455Z     - libgcc 16.2.0 ha9f2e26_7
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6390925Z     - libgomp 16.2.0 he0feb66_7
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6391438Z     - liblzma 5.8.3 hb03c661_1
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6391912Z     - libnsl 2.0.1 hb9d3cd8_1
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6392389Z     - libpython 3.12.14 h0c77377_3_cpython
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6392913Z     - libsqlite 3.53.4 h13e7031_1
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6393396Z     - libstdcxx 16.2.0 h934c35e_7
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6393883Z     - libuuid 2.42.4 hcfc3c73_0
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6394390Z     - libxcrypt 4.4.38 h280c20c_0
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6394953Z     - libzlib 1.3.2 h25fd6f3_3
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6395438Z     - ncurses 6.6 hdb14827_1
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6395898Z     - openssl 3.6.4 h781a0a9_0
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6396481Z     - packaging 26.3 pyhc364b38_0
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6397010Z     - pip 26.2.1 pyh8b19718_0
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6397481Z     - python 3.12.14 h5f976f7_3_cpython
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6397979Z     - readline 8.3 hd6e31c0_1
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6398461Z     - setuptools 84.0.0 pyh332efcf_0
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6398989Z     - tk 8.6.13 noxft_h1df4ec4_4
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6399551Z     - tzdata 2026c h151e31d_0
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6400029Z     - wheel 0.48.0 pyhd8ed1ab_0
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6400501Z     - zstd 1.5.7 hb78ec9c_7
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6400946Z   run:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6401355Z     - numpy >=2.0.0
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6401766Z     - python >=3.12
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6402163Z     - scipy >=1.14.0
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6402566Z     - typing_extensions >=4.0.0
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6403059Z     - warp-lang
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6403470Z test:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6403908Z   requires:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6404281Z     - pytest
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6404723Z     - pip
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6405132Z     - python 3.12.*
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6405574Z   source_files:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6405963Z     - particula
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6406462Z   commands:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6406832Z     - pip check
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6407370Z     - python -c "import particula; print(particula.__version__)"
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6410023Z     - pytest -W error -W "ignore::pytest.PytestUnknownMarkWarning" -m "not slow and
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6410809Z       not performance"
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6411444Z about:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6411885Z   home: https://uncscode.github.io/particula
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6412497Z   license: MIT
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6412896Z   license_family: MIT
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6413302Z   license_file: license
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6413871Z   summary: a simple, fast, and powerful particle simulator
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6414647Z   description: 'Particula is a Python-based aerosol particle simulator. Its goal is
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6415390Z     to
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6415635Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6416027Z     provide a robust aerosol simulation (including both gas and particle phases)
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6416839Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6417251Z     that can be used to answer scientific questions arising from experiments and research
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6417994Z     endeavors.
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6418200Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6418382Z     '
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6418870Z   dev_url: https://github.com/uncscode/particula
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6419486Z   doc_url: https://uncscode.github.io/particula/
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6419989Z extra:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6420393Z   recipe-maintainers:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6420851Z     - gorkowski
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6421267Z     - ngam
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6421642Z     - mahf708
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6422015Z   final: true
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6422437Z   copy_test_source_files: true
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6422883Z ```
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6423129Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6423489Z source tree in: /home/conda/feedstock_root/build_artifacts/particula_1791096598735/work
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6671302Z export PREFIX=/home/conda/feedstock_root/build_artifacts/particula_1791096598735/_h_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_p
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6673450Z export BUILD_PREFIX=/home/conda/feedstock_root/build_artifacts/particula_1791096598735/_build_env
linux_64_	UNKNOWN STEP	2026-10-04T06:51:18.6674606Z export SRC_DIR=/home/conda/feedstock_root/build_artifacts/particula_1791096598735/work
linux_64_	UNKNOWN STEP	2026-10-04T06:51:19.9601464Z Using pip 26.2.1 from $PREFIX/lib/python3.12/site-packages/pip (python 3.12)
linux_64_	UNKNOWN STEP	2026-10-04T06:51:19.9602031Z Non-user install because user site-packages disabled
linux_64_	UNKNOWN STEP	2026-10-04T06:51:19.9751129Z Ignoring indexes: https://pypi.org/simple
linux_64_	UNKNOWN STEP	2026-10-04T06:51:19.9755605Z Created temporary directory: /tmp/pip-build-tracker-2gk4_u31
linux_64_	UNKNOWN STEP	2026-10-04T06:51:19.9756803Z Initialized build tracking at /tmp/pip-build-tracker-2gk4_u31
linux_64_	UNKNOWN STEP	2026-10-04T06:51:19.9768614Z Created build tracker: /tmp/pip-build-tracker-2gk4_u31
linux_64_	UNKNOWN STEP	2026-10-04T06:51:19.9769278Z Entered build tracker: /tmp/pip-build-tracker-2gk4_u31
linux_64_	UNKNOWN STEP	2026-10-04T06:51:19.9769730Z Created temporary directory: /tmp/pip-install-syf8323q
linux_64_	UNKNOWN STEP	2026-10-04T06:51:19.9770317Z Created temporary directory: /tmp/pip-ephem-wheel-cache-s4ormbj7
linux_64_	UNKNOWN STEP	2026-10-04T06:51:19.9865503Z Processing ./.
linux_64_	UNKNOWN STEP	2026-10-04T06:51:19.9870425Z   Added file://$SRC_DIR to build tracker '/tmp/pip-build-tracker-2gk4_u31'
linux_64_	UNKNOWN STEP	2026-10-04T06:51:19.9882426Z   Created temporary directory: /tmp/pip-modern-metadata-fm01e9_r
linux_64_	UNKNOWN STEP	2026-10-04T06:51:19.9886645Z   Preparing metadata (pyproject.toml): started
linux_64_	UNKNOWN STEP	2026-10-04T06:51:19.9887856Z   Running command Preparing metadata (pyproject.toml)
linux_64_	UNKNOWN STEP	2026-10-04T06:51:20.0630303Z   Preparing metadata (pyproject.toml): finished with status 'done'
linux_64_	UNKNOWN STEP	2026-10-04T06:51:20.0640359Z   Source in ./. has version 0.2.14, which satisfies requirement particula==0.2.14 from file://$SRC_DIR
linux_64_	UNKNOWN STEP	2026-10-04T06:51:20.0642436Z   Removed particula==0.2.14 from file://$SRC_DIR from build tracker '/tmp/pip-build-tracker-2gk4_u31'
linux_64_	UNKNOWN STEP	2026-10-04T06:51:20.0668231Z Created temporary directory: /tmp/pip-unpack-3wpgy88p
linux_64_	UNKNOWN STEP	2026-10-04T06:51:20.0669325Z Building wheels for collected packages: particula
linux_64_	UNKNOWN STEP	2026-10-04T06:51:20.0675443Z   Destination directory: /tmp/pip-ephem-wheel-cache-s4ormbj7/wheels/d1/aa/20/97b9b9d9be2d92719a79e9be879522699684f20fcca638f24e/tmpsmh797gb
linux_64_	UNKNOWN STEP	2026-10-04T06:51:20.0680589Z   Building wheel for particula (pyproject.toml): started
linux_64_	UNKNOWN STEP	2026-10-04T06:51:20.0682122Z   Running command Building wheel for particula (pyproject.toml)
linux_64_	UNKNOWN STEP	2026-10-04T06:51:20.4541326Z   Building wheel for particula (pyproject.toml): finished with status 'done'
linux_64_	UNKNOWN STEP	2026-10-04T06:51:20.4575029Z   Created wheel for particula: filename=particula-0.2.14-py3-none-any.whl size=2306216 sha256=f5a89db284aefd0d86c157821f0687d939e22006347daf7f35b0d17a3d19d4bb
linux_64_	UNKNOWN STEP	2026-10-04T06:51:20.4576328Z   Stored in directory: /tmp/pip-ephem-wheel-cache-s4ormbj7/wheels/d1/aa/20/97b9b9d9be2d92719a79e9be879522699684f20fcca638f24e
linux_64_	UNKNOWN STEP	2026-10-04T06:51:20.4616784Z Successfully built particula
linux_64_	UNKNOWN STEP	2026-10-04T06:51:20.4709418Z Installing collected packages: particula
linux_64_	UNKNOWN STEP	2026-10-04T06:51:21.5146855Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:21.5250634Z Successfully installed particula-0.2.14
linux_64_	UNKNOWN STEP	2026-10-04T06:51:21.5251845Z Removed build tracker: '/tmp/pip-build-tracker-2gk4_u31'
linux_64_	UNKNOWN STEP	2026-10-04T06:51:22.7111355Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:22.7112554Z Resource usage statistics from building particula:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:22.7113478Z    Process count: 3
linux_64_	UNKNOWN STEP	2026-10-04T06:51:22.7114518Z    CPU time: Sys=0:00:00.1, User=0:00:00.4
linux_64_	UNKNOWN STEP	2026-10-04T06:51:22.7115366Z    Memory: 70.4M
linux_64_	UNKNOWN STEP	2026-10-04T06:51:22.7115831Z    Disk usage: 31.8K
linux_64_	UNKNOWN STEP	2026-10-04T06:51:22.7116504Z    Time elapsed: 0:00:04.0
linux_64_	UNKNOWN STEP	2026-10-04T06:51:22.7116882Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:22.7116895Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:23.0480828Z Packaging particula
linux_64_	UNKNOWN STEP	2026-10-04T06:51:23.4257117Z Packaging particula-0.2.14-pyhd8ed1ab_0
linux_64_	UNKNOWN STEP	2026-10-04T06:51:24.3764274Z number of files: 536
linux_64_	UNKNOWN STEP	2026-10-04T06:51:24.5375249Z Fixing permissions
linux_64_	UNKNOWN STEP	2026-10-04T06:51:24.5510784Z Adding the following extra-meta data to about.json: {'flow_run_id': 'github_37183899557', 'remote_url': 'https://github.com/conda-forge/particula-feedstock', 'sha': '6a261a2433a30adb87c2a63adc84befc43e1846b'}
linux_64_	UNKNOWN STEP	2026-10-04T06:51:24.8559650Z Packaged license file/s.
linux_64_	UNKNOWN STEP	2026-10-04T06:51:24.9541669Z INFO :: Time taken to mark (prefix)
linux_64_	UNKNOWN STEP	2026-10-04T06:51:24.9542227Z         0 replacements in 0 files was 0.06 seconds
linux_64_	UNKNOWN STEP	2026-10-04T06:51:33.2421855Z TEST START: /home/conda/feedstock_root/build_artifacts/noarch/particula-0.2.14-pyhd8ed1ab_0.conda
linux_64_	UNKNOWN STEP	2026-10-04T06:51:33.4491601Z Adding in variants from /tmp/tmp47q1zuuc/info/recipe/conda_build_config.yaml
linux_64_	UNKNOWN STEP	2026-10-04T06:51:33.4492628Z Adding in variants from config.variant
linux_64_	UNKNOWN STEP	2026-10-04T06:51:33.5557079Z Renaming /home/conda/feedstock_root/build_artifacts/particula_1791096598735/_build_env prefix directory '/home/conda/feedstock_root/build_artifacts/particula_1791096598735/_build_env' to '/home/conda/feedstock_root/build_artifacts/particula_1791096598735/build_prefix_moved_particula-0.2.14-pyhd8ed1ab_0_linux-64'
linux_64_	UNKNOWN STEP	2026-10-04T06:51:33.5560661Z shutil.move(/home/conda/feedstock_root/build_artifacts/particula_1791096598735/_build_env prefix)=/home/conda/feedstock_root/build_artifacts/particula_1791096598735/_build_env, dest=/home/conda/feedstock_root/build_artifacts/particula_1791096598735/build_prefix_moved_particula-0.2.14-pyhd8ed1ab_0_linux-64)
linux_64_	UNKNOWN STEP	2026-10-04T06:51:33.5563889Z Renaming work directory '/home/conda/feedstock_root/build_artifacts/particula_1791096598735/work' to '/home/conda/feedstock_root/build_artifacts/particula_1791096598735/work_moved_particula-0.2.14-pyhd8ed1ab_0_noarch'
linux_64_	UNKNOWN STEP	2026-10-04T06:51:33.5566499Z shutil.move(work)=/home/conda/feedstock_root/build_artifacts/particula_1791096598735/work, dest=/home/conda/feedstock_root/build_artifacts/particula_1791096598735/work_moved_particula-0.2.14-pyhd8ed1ab_0_noarch)
linux_64_	UNKNOWN STEP	2026-10-04T06:51:37.3714844Z Reloading output folder: ...working... done
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.8968855Z Solving environment (_test_env): ...working... done
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9752588Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9753471Z ## Package Plan ##
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9753855Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9755094Z   environment location: /home/conda/feedstock_root/build_artifacts/particula_1791096598735/_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9756811Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9758622Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9759103Z The following NEW packages will be INSTALLED:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9759624Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9759910Z     _openmp_mutex:         4.5-20_gnu                   conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9760536Z     bzip2:                 1.0.8-hda65f42_10            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9761013Z     c-ares:                1.34.8-hebe6cf0_2            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9761449Z     ca-certificates:       2026.7.22-hbd8a1cb_0         conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9761880Z     colorama:              0.4.6-pyhd8ed1ab_1           conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9762362Z     exceptiongroup:        1.3.1-pyhd8ed1ab_0           conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9763016Z     flatbuffers:           25.9.23-hb7d4c21_0           conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9763446Z     icu:                   78.3-py310h44b86e0_2         conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9763880Z     importlib-metadata:    9.0.1-pyhcf101f3_0           conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9764296Z     iniconfig:             2.3.0-pyhd8ed1ab_0           conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9764729Z     jax:                   0.10.2-pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9765156Z     jaxlib:                0.10.2-cpu_py312h02fec33_1   conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9765576Z     ld_impl_linux-64:      2.46.1-default_hbd61a6d_102  conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9766050Z     libabseil:             20260107.1-cxx17_h7b12aa8_0  conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9766633Z     libblas:               3.11.0-11_h4a7cf45_openblas  conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9767146Z     libcblas:              3.11.0-11_h0358290_openblas  conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9767632Z     libexpat:              2.8.5-hd2095e1_0             conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9768095Z     libffi:                3.7.0-h81df57d_1             conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9768555Z     libgcc:                16.2.0-ha9f2e26_7            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9769003Z     libgfortran:           16.2.0-h69a702a_7            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9769934Z     libgfortran5:          16.2.0-h6b99dfc_7            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9770419Z     libgomp:               16.2.0-he0feb66_7            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9770841Z     libgrpc:               1.78.1-h1d1128b_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9771270Z     liblapack:             3.11.0-11_h47877c9_openblas  conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9771689Z     liblzma:               5.8.3-hb03c661_1             conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9772103Z     libnsl:                2.0.1-hb9d3cd8_1             conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9772542Z     libopenblas:           0.3.34-pthreads_hf13c14d_2   conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9773056Z     libprotobuf:           6.33.5-h538a264_2            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9773505Z     libpython:             3.12.14-h0c77377_3_cpython   conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9773966Z     libre2-11:             2025.11.05-h0dc7533_1        conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9774415Z     libsqlite:             3.53.4-h13e7031_1            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9774833Z     libstdcxx:             16.2.0-h934c35e_7            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9775224Z     libuuid:               2.42.4-hcfc3c73_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9775660Z     libxcrypt:             4.4.38-h280c20c_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9776067Z     libzlib:               1.3.2-h25fd6f3_3             conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9776719Z     ml_dtypes:             0.5.4-np2py312h748b537_2     conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9777197Z     ncurses:               6.6-hdb14827_1               conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9777587Z     numpy:                 2.5.3-py312he827f4e_0        conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9778025Z     onednn:                3.12.1-threadpool_h77e0eb8_0 conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9778467Z     onednn-cpu-threadpool: 3.12.1-threadpool_h2c17ba0_0 conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9778921Z     openssl:               3.6.4-h781a0a9_0             conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9779373Z     opt_einsum:            3.4.0-pyhd8ed1ab_1           conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9779891Z     packaging:             26.3-pyhc364b38_0            conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9780335Z     particula:             0.2.14-pyhd8ed1ab_0          local      
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9780773Z     pip:                   26.2.1-pyh8b19718_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9781195Z     pluggy:                1.6.0-pyhf9edf01_1           conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9781619Z     pygments:              2.21.0-pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9782028Z     pytest:                9.1.1-pyhc364b38_2           conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9782518Z     python:                3.12.14-h5f976f7_3_cpython   conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9782929Z     python_abi:            3.12-9_cp312                 conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9783331Z     re2:                   2025.11.05-h5301d42_1        conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9783748Z     readline:              8.3-hd6e31c0_1               conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9784191Z     scipy:                 1.18.1-py312h106c528_1       conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9784619Z     setuptools:            84.0.0-pyh332efcf_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9785028Z     tk:                    8.6.13-noxft_h1df4ec4_4      conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9785428Z     tomli:                 2.4.1-pyhcf101f3_0           conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9785879Z     typing_extensions:     4.16.0-pyhcf101f3_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9786568Z     tzdata:                2026c-h151e31d_0             conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9787000Z     warp-lang:             1.11.1-cpu_ha977375_         conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9787491Z     wheel:                 0.48.0-pyhd8ed1ab_0          conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9787876Z     zipp:                  4.1.1-pyh5ded981_0           conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9788298Z     zstd:                  1.5.7-hb78ec9c_7             conda-forge
linux_64_	UNKNOWN STEP	2026-10-04T06:51:39.9788515Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:42.9883125Z Preparing transaction: ...working... done
linux_64_	UNKNOWN STEP	2026-10-04T06:51:47.7400685Z Verifying transaction: ...working... done
linux_64_	UNKNOWN STEP	2026-10-04T06:51:53.4772754Z WARNING:conda.conda_pypi.main:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:53.4773864Z   Did you know? You can install many PyPI packages with conda
linux_64_	UNKNOWN STEP	2026-10-04T06:51:53.4774590Z Executing transaction: ...working... done
linux_64_	UNKNOWN STEP	2026-10-04T06:51:53.4775258Z   using conda-pypi. Get started:
linux_64_	UNKNOWN STEP	2026-10-04T06:51:53.4775929Z     https://bit.ly/4xXYt0B
linux_64_	UNKNOWN STEP	2026-10-04T06:51:53.4776757Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:51:53.5010042Z export PREFIX=/home/conda/feedstock_root/build_artifacts/particula_1791096598735/_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh
linux_64_	UNKNOWN STEP	2026-10-04T06:51:53.5013135Z export SRC_DIR=/home/conda/feedstock_root/build_artifacts/particula_1791096598735/test_tmp
linux_64_	UNKNOWN STEP	2026-10-04T06:51:54.3396386Z + pip check
linux_64_	UNKNOWN STEP	2026-10-04T06:51:54.5334077Z No broken requirements found.
linux_64_	UNKNOWN STEP	2026-10-04T06:51:54.5642077Z + python -c 'import particula; print(particula.__version__)'
linux_64_	UNKNOWN STEP	2026-10-04T06:51:55.4453432Z 0.2.14
linux_64_	UNKNOWN STEP	2026-10-04T06:51:55.5215987Z + pytest -W error -W ignore::pytest.PytestUnknownMarkWarning -m 'not slow and not performance'
linux_64_	UNKNOWN STEP	2026-10-04T06:51:55.6590190Z ============================= test session starts ==============================
linux_64_	UNKNOWN STEP	2026-10-04T06:51:55.6591993Z platform linux -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0
linux_64_	UNKNOWN STEP	2026-10-04T06:51:55.6592812Z rootdir: $SRC_DIR
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6263379Z collected 6162 items / 57 deselected / 5 skipped / 6105 selected
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6267326Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6323986Z particula/activity/tests/activity_coefficients_test.py .....             [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6345921Z particula/activity/tests/activity_exports_test.py ...                    [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6376787Z particula/activity/tests/bat_blending_test.py .....                      [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6405442Z particula/activity/tests/convert_functional_group_test.py .....          [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6435040Z particula/activity/tests/gibbs_mixing_test.py ..                         [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6455471Z particula/activity/tests/gibbs_test.py ...                               [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6481335Z particula/activity/tests/phase_separation_test.py ....                   [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6520226Z particula/activity/tests/ratio_test.py ......                            [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6580625Z particula/activity/tests/water_activity_test.py ..                       [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6592118Z particula/dynamics/coagulation/coagulation_builder/tests/brownian_coagulation_builder_test.py . [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6605589Z .                                                                        [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6611080Z particula/dynamics/coagulation/coagulation_builder/tests/charged_coagulation_builder_test.py . [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6638363Z ...                                                                      [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6642107Z particula/dynamics/coagulation/coagulation_builder/tests/coagulation_builder_mixin_test.py . [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6686444Z .......                                                                  [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6693004Z particula/dynamics/coagulation/coagulation_builder/tests/combine_coagulation_strategy_builder_test.py . [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6704009Z .                                                                        [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6708886Z particula/dynamics/coagulation/coagulation_builder/tests/sedimentation_coagulation_builder_test.py . [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6721354Z .                                                                        [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6727811Z particula/dynamics/coagulation/coagulation_builder/tests/turbulent_dns_coagulation_builder_test.py . [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6744373Z .                                                                        [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6749599Z particula/dynamics/coagulation/coagulation_builder/tests/turbulent_shear_coagulation_builder_test.py . [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6764605Z .                                                                        [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.6804760Z particula/dynamics/coagulation/coagulation_strategy/tests/brownian_coagulation_strategy_test.py . [  0%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.8422724Z ...........                                                              [  1%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:02.8509711Z particula/dynamics/coagulation/coagulation_strategy/tests/charged_coagulation_strategy_test.py . [  1%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:03.1930493Z .........                                                                [  1%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:03.1934451Z particula/dynamics/coagulation/coagulation_strategy/tests/coagulation_strategy_abc_test.py . [  1%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:03.2265808Z .........................................                                [  1%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:03.2324410Z particula/dynamics/coagulation/coagulation_strategy/tests/combine_coagulation_strategy_test.py . [  1%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:03.2582915Z ..                                                                       [  2%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:03.2620826Z particula/dynamics/coagulation/coagulation_strategy/tests/sedimentation_coagulation_strategy_test.py . [  2%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:03.3897259Z .....                                                                    [  2%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:03.4376489Z particula/dynamics/coagulation/coagulation_strategy/tests/turbulent_dns_coagulation_strategy_test.py . [  2%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:03.7037257Z .....                                                                    [  2%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:03.7074672Z particula/dynamics/coagulation/coagulation_strategy/tests/turbulent_shear_coagulation_strategy_test.py . [  2%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:03.8431617Z .....                                                                    [  2%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:03.8438666Z particula/dynamics/coagulation/particle_resolved_step/tests/particle_resolved_method_test.py . [  2%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1004327Z .............                                                            [  2%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1059935Z particula/dynamics/coagulation/tests/brownian_kernel_test.py ........... [  2%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1086471Z ...                                                                      [  2%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1097029Z particula/dynamics/coagulation/tests/charge_dimensional_kernel_test.py . [  2%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1168354Z ......                                                                   [  2%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1173041Z particula/dynamics/coagulation/tests/charged_dimensionless_kernel_test.py . [  2%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1216536Z .....                                                                    [  2%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1285006Z particula/dynamics/coagulation/tests/charged_kernel_bugs_test.py ....... [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1382671Z .......                                                                  [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1400384Z particula/dynamics/coagulation/tests/charged_kernel_strategy_test.py ... [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1424029Z ...                                                                      [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1457728Z particula/dynamics/coagulation/tests/coagulation_factories_test.py ..... [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1469621Z .                                                                        [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1501507Z particula/dynamics/coagulation/tests/coagulation_rate_test.py ....       [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1520014Z particula/dynamics/coagulation/tests/kernel_interpolation_diagnostic_test.py . [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1625880Z .....                                                                    [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1644307Z particula/dynamics/coagulation/tests/sedimentation_kernel_test.py ..     [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1666728Z particula/dynamics/coagulation/tests/turbulent_shear_kernel_test.py .... [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1672754Z                                                                          [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1681578Z particula/dynamics/coagulation/turbulent_dns_kernel/tests/g12_radial_distribution_ao2008_test.py . [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1700626Z ..                                                                       [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1709153Z particula/dynamics/coagulation/turbulent_dns_kernel/tests/kernel_ao2008_test.py . [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1724649Z .                                                                        [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1730748Z particula/dynamics/coagulation/turbulent_dns_kernel/tests/phi_ao2008_test.py . [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1754387Z ...                                                                      [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1760282Z particula/dynamics/coagulation/turbulent_dns_kernel/tests/psi_ao2008_test.py . [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1784279Z ...                                                                      [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1789440Z particula/dynamics/coagulation/turbulent_dns_kernel/tests/radial_velocity_module_test.py . [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1813895Z ...                                                                      [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1827166Z particula/dynamics/coagulation/turbulent_dns_kernel/tests/sigma_relative_velocity_ao2008_test.py . [  3%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1854719Z ..                                                                       [  4%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1860347Z particula/dynamics/coagulation/turbulent_dns_kernel/tests/velocity_correlation_f2_ao2008_test.py . [  4%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1883403Z ...                                                                      [  4%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1887854Z particula/dynamics/coagulation/turbulent_dns_kernel/tests/velocity_correlation_terms_ao2008_test.py . [  4%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1952042Z ...........                                                              [  4%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1960090Z particula/dynamics/condensation/condensation_builder/tests/condensation_builder_mixin_test.py . [  4%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1975751Z ..                                                                       [  4%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1980985Z particula/dynamics/condensation/condensation_builder/tests/condensation_isothermal_builder_test.py . [  4%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1994044Z .                                                                        [  4%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.1997977Z particula/dynamics/condensation/condensation_builder/tests/condensation_isothermal_staggered_builder_test.py . [  4%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.2059246Z ..........                                                               [  4%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.2076601Z particula/dynamics/condensation/tests/condensation_factories_test.py ... [  4%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.2123984Z ......                                                                   [  4%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.2130328Z particula/dynamics/condensation/tests/condensation_latent_heat_builder_test.py . [  4%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.2276748Z .....................                                                    [  5%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.3015854Z particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py F [  5%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.3483561Z EFEEEEEFF                                                                [  5%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.3510951Z particula/dynamics/condensation/tests/condensation_strategies_test.py .. [  5%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.4750102Z ........................................................................ [  6%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.6385037Z ......................................................................   [  7%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.6457398Z particula/dynamics/condensation/tests/mass_transfer_test.py ............ [  7%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.6755461Z ............................................                             [  8%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.6790548Z particula/dynamics/condensation/tests/mass_transfer_utils_test.py ...... [  8%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.6807348Z ..                                                                       [  8%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.6832705Z particula/dynamics/condensation/tests/staggered_mass_conservation_test.py . [  8%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.8845584Z ..................                                                       [  8%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.8876371Z particula/dynamics/condensation/tests/staggered_performance_test.py .    [  8%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.8932233Z particula/dynamics/nucleation/tests/nucleation_builders_test.py ........ [  9%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.9215787Z ......................................                                   [  9%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.9253819Z particula/dynamics/nucleation/tests/nucleation_factories_test.py ....... [  9%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.9294561Z ....                                                                     [  9%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.9347512Z particula/dynamics/nucleation/tests/nucleation_strategies_test.py ...... [  9%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.9876456Z ........................................................................ [ 11%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:04.9934803Z ........                                                                 [ 11%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:05.0034814Z particula/dynamics/nucleation/tests/particle_source_test.py ............ [ 11%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:05.0681849Z .................................................................        [ 12%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:05.0716066Z particula/dynamics/properties/tests/dyn_wall_loss_test.py ....           [ 12%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:05.0738469Z particula/dynamics/tests/condensation_exports_test.py ....               [ 12%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:05.0752354Z particula/dynamics/tests/dilution_exports_test.py ..                     [ 12%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:05.0923288Z particula/dynamics/tests/dilution_runnable_test.py ..................... [ 13%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:05.1337410Z ...................................                                      [ 13%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:05.1546247Z particula/dynamics/tests/dilution_test.py .............................. [ 14%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:05.2198850Z ........................................................................ [ 15%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:05.2562394Z .............................                                            [ 15%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:05.2587304Z particula/dynamics/tests/dyn_wall_loss_rate_test.py ..                   [ 15%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:05.2760627Z particula/dynamics/tests/nucleation_runnable_test.py ................... [ 16%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:05.3079628Z ..................................                                       [ 16%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:05.3092083Z particula/dynamics/tests/wall_loss_builders_factories_test.py ..         [ 16%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:06.0648346Z particula/dynamics/tests/wall_loss_runnable_test.py ...........          [ 16%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:06.5021215Z particula/dynamics/tests/wall_loss_strategies_test.py ................   [ 17%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:06.5079129Z particula/dynamics/wall_loss/tests/wall_loss_builders_test.py .......... [ 17%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:06.5153064Z .............                                                            [ 17%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:06.5189871Z particula/dynamics/wall_loss/tests/wall_loss_factories_test.py .....     [ 17%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:06.5234593Z particula/equilibria/tests/equilibria_builders_test.py ........          [ 17%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:06.5290472Z particula/equilibria/tests/equilibria_factories_test.py ...........      [ 17%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:06.5397085Z particula/equilibria/tests/equilibria_imports_test.py .........          [ 18%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:06.5778453Z particula/equilibria/tests/equilibria_strategies_test.py ..........      [ 18%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:06.6039480Z particula/equilibria/tests/equilibria_test.py .............              [ 18%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:06.6235376Z particula/equilibria/tests/partitioning_test.py ..........               [ 18%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:06.6386341Z particula/execution/tests/availability_test.py ......................... [ 19%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:07.4386599Z ..............                                                           [ 19%]
linux_64_	UNKNOWN STEP	2026-10-04T06:52:07.4636325Z particula/execution/tests/captured_full_loop_test.py ................... [ 19%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:16.8830410Z ................................ssssssssssss.....                        [ 20%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:19.7903659Z particula/execution/tests/checkpoint_test.py ........................... [ 20%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:21.4775732Z ...............................                                          [ 21%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:21.4885655Z particula/execution/tests/coagulation_adapter_test.py .................. [ 21%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:23.0992662Z .........................................................                [ 22%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:25.9306001Z particula/execution/tests/coagulation_integration_test.py ............ss [ 22%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:25.9310330Z                                                                          [ 22%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:25.9584200Z particula/execution/tests/communication_test.py ........................ [ 23%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:26.7575630Z .......................................                                  [ 23%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:27.5399333Z particula/execution/tests/condensation_adapter_test.py ................. [ 24%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:33.7270421Z .........................................                                [ 24%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:33.7957085Z particula/execution/tests/condensation_integration_test.py ............. [ 24%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:33.8116904Z ...sss                                                                   [ 25%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:33.8764055Z particula/execution/tests/diagnostics_test.py .............              [ 25%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:34.6556554Z particula/execution/tests/errors_test.py .........................       [ 25%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:35.4393492Z particula/execution/tests/exports_test.py ...                            [ 25%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:35.4430124Z particula/execution/tests/fallback_integration_test.py ...               [ 25%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:35.4617769Z particula/execution/tests/fallback_test.py ............................. [ 26%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:36.2208039Z ........                                                                 [ 26%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:36.6295899Z particula/execution/tests/full_loop_test.py .................            [ 26%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:36.7299925Z particula/execution/tests/gpu_resident_session_example_test.py EEEEFEEEE [ 26%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:36.7364069Z F                                                                        [ 26%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:36.8190989Z particula/execution/tests/gpu_resources_test.py ........................ [ 27%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:37.7564436Z ........................................................................ [ 28%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:37.8474154Z ...........................                                              [ 28%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:38.6320050Z particula/execution/tests/gpu_session_test.py .......................... [ 29%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:39.4622221Z ........................................................................ [ 30%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:54.8939104Z ......................................................                   [ 31%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:54.9482601Z particula/execution/tests/graph_capture_test.py ........................ [ 31%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:55.6699227Z ........................................................................ [ 32%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:58.0576635Z .......................................................s................ [ 34%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:58.5974955Z ..........................                                               [ 34%]
linux_64_	UNKNOWN STEP	2026-10-04T06:53:58.6266373Z particula/execution/tests/multi_box_communication_test.py .....          [ 34%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:01.2644723Z particula/execution/tests/multi_box_loop_test.py ......s                 [ 34%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:01.3066037Z particula/execution/tests/process_adapters_test.py ..................... [ 35%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:02.1219700Z .............                                                            [ 35%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:02.1382105Z particula/execution/tests/process_graph_test.py ........................ [ 35%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:03.0861403Z .....................                                                    [ 36%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:03.8871001Z particula/execution/tests/resident_benchmark_cuda_support_test.py ...... [ 36%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:03.9013809Z .......................                                                  [ 36%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:04.6764266Z particula/execution/tests/resident_benchmark_support_test.py ........... [ 36%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:04.7417148Z ........................................................................ [ 37%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:04.7534126Z ...................                                                      [ 38%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:04.9200714Z particula/execution/tests/resident_communication_test.py ............... [ 38%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:04.9206412Z                                                                          [ 38%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:05.2013289Z particula/execution/tests/resident_enqueue_test.py .............         [ 38%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:05.2200585Z particula/execution/tests/restart_loop_test.py ..                        [ 38%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:05.2900630Z particula/execution/tests/rng_invariance_test.py .s.s.s.s.s.s.s.s.s.s... [ 39%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:05.2972075Z .                                                                        [ 39%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:05.3186924Z particula/execution/tests/rng_test.py .................................. [ 39%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:06.0991197Z ..............s..                                                        [ 39%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:06.1178710Z particula/execution/tests/scheduler_test.py ............................ [ 40%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:06.9081817Z .............                                                            [ 40%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.7129876Z particula/execution/tests/state_updates_test.py ........................ [ 40%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.7385703Z ..                                                                       [ 40%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.7901140Z particula/execution/tests/thermodynamic_updates_test.py ................ [ 41%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.7905680Z                                                                          [ 41%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.8209581Z particula/execution/tests/transport_loop_test.py ......                  [ 41%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.8230063Z particula/gas/properties/tests/concentration_function_test.py ..         [ 41%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.8267597Z particula/gas/properties/tests/dynamic_viscosity_test.py ......          [ 41%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.8295566Z particula/gas/properties/tests/fluid_rms_velocity_test.py ....           [ 41%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.8339197Z particula/gas/properties/tests/integral_scale_module_test.py .......     [ 41%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.8363544Z particula/gas/properties/tests/kinematic_viscosity_test.py ....          [ 41%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.8408151Z particula/gas/properties/tests/kolmogorov_module_test.py .......         [ 41%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.8486725Z particula/gas/properties/tests/mean_free_path_test.py .............      [ 42%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.8511872Z particula/gas/properties/tests/normalize_accel_variance_test.py ....     [ 42%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.8533036Z particula/gas/properties/tests/partial_pressure_function_test.py ...     [ 42%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.8580547Z particula/gas/properties/tests/taylor_microscale_module_test.py ........ [ 42%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.8610854Z ....                                                                     [ 42%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.8634582Z particula/gas/properties/tests/thermal_conductivity_test.py ....         [ 42%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.8653270Z particula/gas/tests/atmosphere_builder_test.py ..                        [ 42%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.8692813Z particula/gas/tests/atmosphere_test.py ....                              [ 42%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.8846091Z particula/gas/tests/environment_data_test.py ........................... [ 42%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.8954385Z .................                                                        [ 43%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.9131135Z particula/gas/tests/gas_data_test.py ............................        [ 43%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.9286482Z particula/gas/tests/latent_heat_builders_test.py ...............sssssss. [ 44%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.9292055Z                                                                          [ 44%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.9355341Z particula/gas/tests/latent_heat_factories_test.py ...........            [ 44%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.9474551Z particula/gas/tests/latent_heat_strategies_test.py ..................... [ 44%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.9568660Z ...............                                                          [ 44%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.9636681Z particula/gas/tests/species_builder_test.py ..........                   [ 44%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.9724919Z particula/gas/tests/species_facade_test.py .............                 [ 45%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.9761249Z particula/gas/tests/species_factory_test.py ....                         [ 45%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.9829892Z particula/gas/tests/species_private_test.py ..........                   [ 45%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:07.9931306Z particula/gas/tests/species_test.py ..............                       [ 45%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:08.0032826Z particula/gas/tests/vapor_pressure_builders_test.py .................... [ 45%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:08.0059954Z ....                                                                     [ 46%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:08.0103124Z particula/gas/tests/vapor_pressure_factories_test.py .......             [ 46%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:08.0144995Z particula/gas/tests/vapor_pressure_strategies_test.py .......            [ 46%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:24.5035366Z particula/gpu/dynamics/tests/coagulation_funcs_test.py ................. [ 46%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:33.8675986Z ........................................................................ [ 47%]
linux_64_	UNKNOWN STEP	2026-10-04T06:54:44.4132993Z ......................................                                   [ 48%]
linux_64_	UNKNOWN STEP	2026-10-04T06:55:03.2598482Z particula/gpu/dynamics/tests/condensation_funcs_test.py ................ [ 48%]
linux_64_	UNKNOWN STEP	2026-10-04T06:55:05.5941423Z ..                                                                       [ 48%]
linux_64_	UNKNOWN STEP	2026-10-04T06:55:05.5949293Z particula/gpu/dynamics/tests/export_test.py .                            [ 48%]
linux_64_	UNKNOWN STEP	2026-10-04T06:55:21.0350245Z particula/gpu/dynamics/tests/wall_loss_funcs_test.py ................    [ 48%]
linux_64_	UNKNOWN STEP	2026-10-04T06:55:21.1511185Z particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py .. [ 48%]
linux_64_	UNKNOWN STEP	2026-10-04T06:55:37.0068197Z .......                                                                  [ 49%]
linux_64_	UNKNOWN STEP	2026-10-04T06:55:37.0212759Z particula/gpu/kernels/tests/coagulation_test.py ........................ [ 49%]
linux_64_	UNKNOWN STEP	2026-10-04T06:55:42.5513852Z ........................................................................ [ 50%]
linux_64_	UNKNOWN STEP	2026-10-04T06:55:55.8390099Z ........................................................................ [ 51%]
linux_64_	UNKNOWN STEP	2026-10-04T06:55:55.9191383Z ......................................................................s. [ 53%]
linux_64_	UNKNOWN STEP	2026-10-04T06:55:58.9882658Z .................................................................s.s.... [ 54%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:12.5870062Z .........................................................s.............. [ 55%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:30.4619756Z ..........sss............s.........................                      [ 56%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:30.4717263Z particula/gpu/kernels/tests/coagulation_validation_test.py ............. [ 56%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:33.2055842Z ........................................................................ [ 57%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:33.2631499Z .............................                                            [ 58%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:33.2957450Z particula/gpu/kernels/tests/communication_test.py ...................... [ 58%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:33.3682780Z s......................................................                  [ 59%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:33.3780480Z particula/gpu/kernels/tests/condensation_autodiff_test.py ..s...         [ 59%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:33.3878815Z particula/gpu/kernels/tests/condensation_graph_capture_test.py ....ss    [ 59%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:33.4755625Z particula/gpu/kernels/tests/condensation_stiffness_test.py s..........   [ 59%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:36.3266761Z particula/gpu/kernels/tests/condensation_test.py .ssssssss.............. [ 60%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:36.5581673Z .........s..ssssssssssssssss..ss......................s.....ssss........ [ 61%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:36.7855425Z .s.s..........................................s....s..............s..... [ 62%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:42.6378033Z .........s.s.......s.s.s.........................s...................... [ 63%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:44.3536557Z ..............ss...............s.s..........                             [ 64%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:44.3844525Z particula/gpu/kernels/tests/dilution_test.py ...................sss..... [ 64%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:44.4592359Z ............sss......................................................... [ 65%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:44.4969899Z ...................sss......                                             [ 66%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:44.5164621Z particula/gpu/kernels/tests/environment_test.py ...............s........ [ 66%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:44.5242845Z ....s....                                                                [ 66%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:44.6017303Z particula/gpu/kernels/tests/exhaustion_test.py ......................... [ 67%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:44.6371442Z s.......s..s...s                                                         [ 67%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:44.7876978Z particula/gpu/kernels/tests/nucleation_parity_test.py ............ssssss [ 67%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:44.9448458Z ssssss...sss.s.s.s.s...sss.s..ss.s.s                                     [ 68%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:48.4037724Z particula/gpu/kernels/tests/nucleation_test.py ......................... [ 68%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:50.3846913Z ........................................................................ [ 70%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:52.1935512Z .......................................                                  [ 70%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:52.2200877Z particula/gpu/kernels/tests/slot_management_test.py .................... [ 71%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:52.2895134Z ................................ssss...................s                 [ 72%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:52.3136731Z particula/gpu/kernels/tests/thermodynamics_test.py ..................s.. [ 72%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:52.3313980Z ...................                                                      [ 72%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:54.5199684Z particula/gpu/kernels/tests/wall_loss_parity_test.py ................... [ 72%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:55.0434387Z ...................                                                      [ 73%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:55.0651837Z particula/gpu/kernels/tests/wall_loss_test.py .........................s [ 73%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:56.7790224Z ........................................................................ [ 74%]
linux_64_	UNKNOWN STEP	2026-10-04T06:56:59.7112566Z ......s.........................s                                        [ 75%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:04.7377966Z particula/gpu/properties/tests/gas_properties_test.py ...                [ 75%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:23.0543478Z particula/gpu/properties/tests/particle_properties_test.py ............. [ 75%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:34.6596658Z .....                                                                    [ 75%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:34.9920046Z particula/gpu/tests/benchmark_helpers_test.py .......................... [ 76%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:35.2772812Z .............................................................            [ 77%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:35.2948713Z particula/gpu/tests/communication_parity_test.py ......                  [ 77%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:35.3221151Z particula/gpu/tests/conversion_test.py ................................. [ 77%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:35.3742721Z ...............................................................          [ 78%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:35.3779685Z particula/gpu/tests/cuda_availability_test.py ........                   [ 79%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:35.5284897Z particula/gpu/tests/data_containers_example_test.py EEEEEEEFFFEEFE       [ 79%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:35.6305323Z particula/gpu/tests/gpu_coagulation_direct_example_test.py EFEEEEEEEEE   [ 79%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:35.7253165Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py EEFEEE [ 79%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:35.9083642Z EEEEEEEEEEEEEEEE                                                         [ 79%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:36.0065665Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py EEEEEEEE [ 79%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:36.1891016Z EEEEEEEEEEEEEFE                                                          [ 80%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:36.2867772Z particula/gpu/tests/gpu_direct_kernels_example_test.py EEFE...FEEEE      [ 80%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:36.3236976Z particula/gpu/tests/gpu_direct_nucleation_example_test.py FF.F           [ 80%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:38.1636982Z particula/gpu/tests/kernel_exports_test.py .......................       [ 80%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:38.1801115Z particula/gpu/tests/mass_precision_cases_test.py ....................    [ 81%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:38.1957045Z particula/gpu/tests/mass_precision_metrics_test.py ..................... [ 81%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:38.2453056Z ........................................................................ [ 82%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:38.2467031Z .                                                                        [ 82%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:38.3144921Z particula/gpu/tests/process_sequence_test.py ........................s.. [ 83%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:38.3907411Z .................                                                        [ 83%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:38.3918993Z particula/gpu/tests/profiling_smoke_test.py ss                           [ 83%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:39.2085854Z particula/gpu/tests/profiling_support_test.py .......................... [ 83%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:39.2495018Z .................................                                        [ 84%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:39.2522930Z particula/gpu/tests/profiling_workload_runner_test.py ....               [ 84%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:39.2708512Z particula/gpu/tests/warp_types_test.py .............................     [ 84%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:39.3927369Z particula/integration_tests/charged_coagulation_comparison_test.py ...   [ 84%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.2145716Z particula/integration_tests/coagulation_integration_test.py .......      [ 85%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.2153345Z particula/integration_tests/condensation_latent_heat_conservation_test.py . [ 85%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.2258198Z .....                                                                    [ 85%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.2356934Z particula/integration_tests/condensation_particle_resolved_test.py .     [ 85%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.2451473Z particula/integration_tests/gpu_thermodynamics_contract_test.py ..s      [ 85%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.2536993Z particula/integration_tests/nucleation_process_test.py ....              [ 85%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.2575267Z particula/integration_tests/quick_start_test.py .                        [ 85%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.2583124Z particula/particles/distribution_strategies/tests/mass_based_moving_bin_test.py . [ 85%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.2678277Z ...............                                                          [ 85%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.2681209Z particula/particles/distribution_strategies/tests/particle_resolved_speciated_mass_test.py . [ 85%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.2844911Z ..........................                                               [ 86%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.2848144Z particula/particles/distribution_strategies/tests/radii_based_moving_bin_test.py . [ 86%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.2933618Z ..............                                                           [ 86%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.2936838Z particula/particles/distribution_strategies/tests/speciated_mass_moving_bin_test.py . [ 86%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3022865Z ..............                                                           [ 86%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3058223Z particula/particles/properties/tests/aerodynamic_mobility_test.py ...... [ 86%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3081123Z ...                                                                      [ 86%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3136549Z particula/particles/properties/tests/aerodynamic_size_test.py .......... [ 86%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3164224Z ....                                                                     [ 86%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3180582Z particula/particles/properties/tests/collision_radius_module_test.py ... [ 86%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3211688Z .....                                                                    [ 87%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3235498Z particula/particles/properties/tests/convert_kappa_volumes_test.py ....  [ 87%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3243221Z particula/particles/properties/tests/convert_mass_concentration_test.py . [ 87%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3469871Z ................................                                         [ 87%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3503478Z particula/particles/properties/tests/convert_mole_fraction_test.py ..... [ 87%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3514722Z .                                                                        [ 87%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3543287Z particula/particles/properties/tests/coulomb_enhancement_test.py ....    [ 87%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3563120Z particula/particles/properties/tests/diffusion_coefficient_test.py ...   [ 87%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3597609Z particula/particles/properties/tests/diffusive_knudsen_test.py .....     [ 87%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3627496Z particula/particles/properties/tests/friction_factor_test.py .....       [ 88%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3650559Z particula/particles/properties/tests/inertia_time_test.py ...            [ 88%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3704579Z particula/particles/properties/tests/kelvin_effect_test.py ........      [ 88%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3738596Z particula/particles/properties/tests/knudsen_number_test.py ......       [ 88%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3758788Z particula/particles/properties/tests/mean_thermal_speed_test.py ...      [ 88%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3795472Z particula/particles/properties/tests/mixing_state_index_test.py ......   [ 88%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3809429Z particula/particles/properties/tests/organic_density_module_test.py ..   [ 88%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3835976Z particula/particles/properties/tests/reynolds_number_test.py ....        [ 88%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3895094Z particula/particles/properties/tests/settling_velocity_test.py ......... [ 88%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3947438Z ....                                                                     [ 88%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3952760Z particula/particles/properties/tests/size_distribution_convert_test.py . [ 88%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.3982584Z ....                                                                     [ 88%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.4020847Z particula/particles/properties/tests/slip_correction_test.py ......      [ 88%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.4048347Z particula/particles/properties/tests/sorted_bins_test.py ....            [ 88%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.4078984Z particula/particles/properties/tests/special_functions_test.py ...       [ 89%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.4100415Z particula/particles/properties/tests/stokes_number_test.py ...           [ 89%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.4132695Z particula/particles/properties/tests/vapor_correction_test.py .....      [ 89%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.4249915Z particula/particles/tests/activity_builders_test.py ..................   [ 89%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.4312303Z particula/particles/tests/activity_factories_test.py .........           [ 89%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.4447518Z particula/particles/tests/activity_strategies_test.py ................   [ 89%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.4570931Z particula/particles/tests/change_particle_representation_test.py ...     [ 89%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.4592560Z particula/particles/tests/distribution_builders_test.py ....             [ 89%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.4619030Z particula/particles/tests/distribution_facotries_test.py .....           [ 90%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.4819432Z particula/particles/tests/exhaustion_test.py ........................... [ 90%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.7564047Z .........................................................                [ 91%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.7695045Z particula/particles/tests/particle_data_test.py ........................ [ 91%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.7814711Z .............                                                            [ 92%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.7911703Z particula/particles/tests/representation_builders_test.py .........      [ 92%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.8034693Z particula/particles/tests/representation_facade_test.py ................ [ 92%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.8052134Z .                                                                        [ 92%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.8147438Z particula/particles/tests/representation_factories_test.py ......        [ 92%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.8294030Z particula/particles/tests/representation_test.py ....................... [ 92%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.8339385Z ......                                                                   [ 93%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.8349945Z particula/particles/tests/representation_zero_particle_test.py .         [ 93%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.8510306Z particula/particles/tests/slot_management_test.py ...................... [ 93%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.8973880Z ..........................................................               [ 94%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.9025850Z particula/particles/tests/surface_builders_test.py ........              [ 94%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.9055496Z particula/particles/tests/surface_factories_test.py ....                 [ 94%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.9157272Z particula/particles/tests/surface_strategies_test.py .............       [ 94%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.9188533Z particula/tests/abc_builder_test.py .....                                [ 94%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.9378611Z particula/tests/aerosol_builder_test.py ......                           [ 94%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.9431024Z particula/tests/aerosol_test.py ...                                      [ 95%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.9735055Z particula/tests/backend_selected_coagulation_example_test.py F           [ 95%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.9803007Z particula/tests/benchmark_option_test.py ..........                      [ 95%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:41.9881858Z particula/tests/builder_mixin_test.py ............                       [ 95%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:42.0464491Z particula/tests/dilution_example_test.py FFFFFF                          [ 95%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:43.6162879Z particula/tests/execution_exports_test.py ...............                [ 95%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:43.6413854Z particula/tests/execution_test.py ...................................... [ 96%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:45.2396012Z ........................................................................ [ 97%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.0380536Z ..........................                                               [ 97%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.3250362Z particula/tests/gpu_resident_graph_capture_docs_test.py FFFFFFFFFFFFFFFF [ 98%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.3444707Z Fs                                                                       [ 98%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.5601084Z particula/tests/gpu_resident_multi_timestep_docs_test.py EEEEFFFEEEEEEF  [ 98%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.5618859Z particula/tests/import_test.py ...                                       [ 98%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.5630802Z particula/tests/logging_setup_test.py .                                  [ 98%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6000161Z particula/tests/nucleation_example_test.py FFF.                          [ 98%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6206903Z particula/tests/pytest_marker_policy_test.py .F.......                   [ 98%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6221258Z particula/tests/runnable_test.py .                                       [ 98%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6240902Z particula/util/chemical/tests/chemical_properties_test.py ss.            [ 98%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6256506Z particula/util/chemical/tests/chemical_search_test.py ss.                [ 98%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6274206Z particula/util/chemical/tests/surface_tension_test.py ss.                [ 98%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6290253Z particula/util/chemical/tests/vapor_pressure_test.py ss.                 [ 98%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6341237Z particula/util/lf2013_coagulation/tests/src_lf2013_coagulation_test.py . [ 99%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6388590Z .                                                                        [ 99%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6398451Z particula/util/tests/arbitrary_round_test.py .                           [ 99%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6421837Z particula/util/tests/colors_test.py ....                                 [ 99%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6429781Z particula/util/tests/constants_test.py .                                 [ 99%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6475222Z particula/util/tests/convert_dtypes_test.py ........                     [ 99%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6523845Z particula/util/tests/machine_limit_test.py .......                       [ 99%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6583793Z particula/util/tests/reduced_quantity_test.py ...........                [ 99%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6610024Z particula/util/tests/refractive_index_test.py .....                      [ 99%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6627583Z particula/util/tests/util_init_test.py ...                               [ 99%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6819829Z particula/util/tests/validate_inputs_test.py ....................        [100%]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6837062Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6837794Z ==================================== ERRORS ====================================
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6839036Z _ ERROR at setup of test_condensation_latent_heat_run_example_returns_finite_structured_results _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6839728Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6840219Z     @pytest.fixture(scope="module")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6841054Z     def example_namespace() -> ExampleNamespace:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6841997Z         """Load the published example module once for runtime assertions."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6843072Z >       return cast(ExampleNamespace, runpy.run_path(str(EXAMPLE_PATH)))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6845359Z                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6845945Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6847776Z particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py:51: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6849432Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6852091Z <frozen runpy>:286: in run_path
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6853985Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6854891Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6855456Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6856233Z fname = '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6856983Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6857320Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6859085Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6864347Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6864852Z <frozen runpy>:254: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6865835Z _ ERROR at setup of test_condensation_latent_heat_as_float_uses_first_scalar_value _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6866813Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6867382Z     @pytest.fixture(scope="module")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6869669Z     def example_namespace() -> ExampleNamespace:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6870573Z         """Load the published example module once for runtime assertions."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6871714Z >       return cast(ExampleNamespace, runpy.run_path(str(EXAMPLE_PATH)))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6872589Z                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6873103Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6873706Z particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py:51: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6874741Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6875519Z <frozen runpy>:286: in run_path
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6876384Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6877434Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6878017Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6878621Z fname = '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6879357Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6879635Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6880679Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6881792Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6882171Z <frozen runpy>:254: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6883195Z _ ERROR at setup of test_condensation_latent_heat_build_aerosol_creates_single_box_state _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6883889Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6884295Z     @pytest.fixture(scope="module")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6885031Z     def example_namespace() -> ExampleNamespace:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6885913Z         """Load the published example module once for runtime assertions."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6887054Z >       return cast(ExampleNamespace, runpy.run_path(str(EXAMPLE_PATH)))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6887984Z                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6888489Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6889124Z particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py:51: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6890132Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6890917Z <frozen runpy>:286: in run_path
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6891583Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6892224Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6892798Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6893534Z fname = '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6894174Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6894484Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6895536Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6896683Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6897005Z <frozen runpy>:254: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6898386Z _ ERROR at setup of test_condensation_latent_heat_main_path_matches_run_example_contract _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6899095Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6899506Z     @pytest.fixture(scope="module")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6900244Z     def example_namespace() -> ExampleNamespace:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6901152Z         """Load the published example module once for runtime assertions."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6902353Z >       return cast(ExampleNamespace, runpy.run_path(str(EXAMPLE_PATH)))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6903231Z                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6903792Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6904435Z particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py:51: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6905484Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6906327Z <frozen runpy>:286: in run_path
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6907019Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6907657Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6908370Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6908967Z fname = '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6909682Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6910003Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6911091Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6912085Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6912416Z <frozen runpy>:254: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6913440Z _ ERROR at setup of test_condensation_latent_heat_example_reports_condensation_or_explicit_zero_transfer _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6914306Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6914720Z     @pytest.fixture(scope="module")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6915458Z     def example_namespace() -> ExampleNamespace:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6916620Z         """Load the published example module once for runtime assertions."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6917723Z >       return cast(ExampleNamespace, runpy.run_path(str(EXAMPLE_PATH)))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6918581Z                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6919142Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6919950Z particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py:51: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6921042Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6921822Z <frozen runpy>:286: in run_path
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6922431Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6923041Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6923612Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6924211Z fname = '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6924855Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6925170Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6926290Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6927278Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6927598Z <frozen runpy>:254: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6928541Z _ ERROR at setup of test_condensation_latent_heat_example_energy_matches_mass_transfer_contract _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6929359Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6929729Z     @pytest.fixture(scope="module")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6930448Z     def example_namespace() -> ExampleNamespace:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6931294Z         """Load the published example module once for runtime assertions."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6932268Z >       return cast(ExampleNamespace, runpy.run_path(str(EXAMPLE_PATH)))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6933099Z                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6933668Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6934274Z particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py:51: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6935247Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6936635Z <frozen runpy>:286: in run_path
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6937262Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6937864Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6938483Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6939051Z fname = '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6939711Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6940027Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6941970Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6942972Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6943469Z <frozen runpy>:254: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6944331Z ________ ERROR at setup of test_forced_disable_skips_loader_and_fixture ________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6945032Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6945559Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea4cf1550>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6946341Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6946678Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6947363Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6948226Z         """Import a fresh example while proving import-time GPU safety."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6948940Z         blocked = {
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6949604Z             "warp",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6950209Z             "particula.gpu",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6950835Z             "particula.execution.gpu_session",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6951503Z             "particula.execution.gpu_resources",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6952170Z             "particula.execution.checkpoint",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6952842Z         }
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6953415Z         original_import = builtins.__import__
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6954031Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6955028Z         def guarded_import(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6955767Z             name: str,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6956623Z             globals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6957384Z             locals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6958097Z             fromlist: Sequence[str] = (),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6958764Z             level: int = 0,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6959348Z         ) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6959999Z             if name in blocked:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6960699Z                 pytest.fail(f"example imported {name} eagerly")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6961574Z             return original_import(name, globals, locals, fromlist, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6962355Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6962972Z         monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6963821Z         sys.modules.pop("gpu_resident_session", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6964651Z         monkeypatch.setattr(builtins, "__import__", guarded_import)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6965588Z >       module = importlib.import_module("gpu_resident_session")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6966664Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6967141Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6967846Z particula/execution/tests/gpu_resident_session_example_test.py:53: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6968784Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6970711Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6972670Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6973665Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6974424Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6975457Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6976673Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6977403Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6977876Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6978332Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.6978918Z name = 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7108619Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7108871Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7108962Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7109255Z E   ModuleNotFoundError: No module named 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7109654Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7109860Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7110334Z ______________ ERROR at setup of test_missing_warp_skips_fixture _______________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7110598Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7110901Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea4b07050>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7136815Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7149921Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7150463Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7151110Z         """Import a fresh example while proving import-time GPU safety."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7151612Z         blocked = {
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7151906Z             "warp",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7152233Z             "particula.gpu",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7152616Z             "particula.execution.gpu_session",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7153058Z             "particula.execution.gpu_resources",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7153547Z             "particula.execution.checkpoint",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7154138Z         }
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7154472Z         original_import = builtins.__import__
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7154888Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7155185Z         def guarded_import(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7155540Z             name: str,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7155913Z             globals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7156589Z             locals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7157954Z             fromlist: Sequence[str] = (),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7158360Z             level: int = 0,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7158687Z         ) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7159147Z             if name in blocked:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7160635Z                 pytest.fail(f"example imported {name} eagerly")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7161436Z             return original_import(name, globals, locals, fromlist, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7161880Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7163055Z         monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7163798Z         sys.modules.pop("gpu_resident_session", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7164590Z         monkeypatch.setattr(builtins, "__import__", guarded_import)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7165315Z >       module = importlib.import_module("gpu_resident_session")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7165868Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7166280Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7166594Z particula/execution/tests/gpu_resident_session_example_test.py:53: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7167218Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7169597Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7171439Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7172072Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7172689Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7173169Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7173511Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7174018Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7174362Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7174712Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7174863Z name = 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7175267Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7175578Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7175678Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7176058Z E   ModuleNotFoundError: No module named 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7176531Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7176690Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7177097Z _____ ERROR at setup of test_broken_enabled_warp_import_propagates[error0] _____
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7177362Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7177563Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea4b05bb0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7177834Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7177934Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7178226Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7178630Z         """Import a fresh example while proving import-time GPU safety."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7178961Z         blocked = {
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7179167Z             "warp",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7179528Z             "particula.gpu",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7179809Z             "particula.execution.gpu_session",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7180119Z             "particula.execution.gpu_resources",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7180429Z             "particula.execution.checkpoint",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7180708Z         }
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7180925Z         original_import = builtins.__import__
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7181180Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7181378Z         def guarded_import(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7181607Z             name: str,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7181861Z             globals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7182266Z             locals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7182571Z             fromlist: Sequence[str] = (),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7182831Z             level: int = 0,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7183052Z         ) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7183261Z             if name in blocked:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7183541Z                 pytest.fail(f"example imported {name} eagerly")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7183924Z             return original_import(name, globals, locals, fromlist, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7184242Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7184487Z         monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7184827Z         sys.modules.pop("gpu_resident_session", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7185188Z         monkeypatch.setattr(builtins, "__import__", guarded_import)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7185583Z >       module = importlib.import_module("gpu_resident_session")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7185917Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7186104Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7186940Z particula/execution/tests/gpu_resident_session_example_test.py:53: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7187656Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7189063Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7189988Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7190328Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7190651Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7190934Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7191170Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7191447Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7191683Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7191912Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7192009Z name = 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7192288Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7192495Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7192570Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7192836Z E   ModuleNotFoundError: No module named 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7193082Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7193233Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7193636Z _____ ERROR at setup of test_broken_enabled_warp_import_propagates[error1] _____
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7193901Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7194101Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea4b06db0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7194372Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7194463Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7194747Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7195151Z         """Import a fresh example while proving import-time GPU safety."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7195485Z         blocked = {
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7195698Z             "warp",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7195919Z             "particula.gpu",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7196458Z             "particula.execution.gpu_session",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7196783Z             "particula.execution.gpu_resources",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7197087Z             "particula.execution.checkpoint",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7197353Z         }
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7197753Z         original_import = builtins.__import__
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7198150Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7198402Z         def guarded_import(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7198723Z             name: str,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7199068Z             globals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7199562Z             locals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7200009Z             fromlist: Sequence[str] = (),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7200395Z             level: int = 0,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7200714Z         ) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7201015Z             if name in blocked:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7201436Z                 pytest.fail(f"example imported {name} eagerly")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7202138Z             return original_import(name, globals, locals, fromlist, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7202594Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7202833Z         monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7203222Z         sys.modules.pop("gpu_resident_session", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7203639Z         monkeypatch.setattr(builtins, "__import__", guarded_import)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7204014Z >       module = importlib.import_module("gpu_resident_session")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7204343Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7204531Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7204705Z particula/execution/tests/gpu_resident_session_example_test.py:53: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7205081Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7205966Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7207752Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7208266Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7208727Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7209015Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7209241Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7209516Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7209746Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7209963Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7210063Z name = 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7210333Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7210524Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7210600Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7210855Z E   ModuleNotFoundError: No module named 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7211091Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7211228Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7211624Z __ ERROR at setup of test_loader_orders_concrete_imports_without_gpu_package ___
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7211895Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7212086Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea4b00050>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7212353Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7212438Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7212722Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7213111Z         """Import a fresh example while proving import-time GPU safety."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7213431Z         blocked = {
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7213630Z             "warp",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7213840Z             "particula.gpu",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7214096Z             "particula.execution.gpu_session",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7214392Z             "particula.execution.gpu_resources",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7214690Z             "particula.execution.checkpoint",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7214946Z         }
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7215150Z         original_import = builtins.__import__
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7215405Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7215588Z         def guarded_import(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7215820Z             name: str,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7216076Z             globals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7216590Z             locals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7217034Z             fromlist: Sequence[str] = (),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7217295Z             level: int = 0,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7217509Z         ) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7217712Z             if name in blocked:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7217984Z                 pytest.fail(f"example imported {name} eagerly")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7218354Z             return original_import(name, globals, locals, fromlist, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7218669Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7218900Z         monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7219302Z         sys.modules.pop("gpu_resident_session", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7220000Z         monkeypatch.setattr(builtins, "__import__", guarded_import)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7220608Z >       module = importlib.import_module("gpu_resident_session")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7221126Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7221416Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7221686Z particula/execution/tests/gpu_resident_session_example_test.py:53: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7222271Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7223437Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7224922Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7225249Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7225546Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7225818Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7226041Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7226523Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7226798Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7227196Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7227354Z name = 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7227922Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7228330Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7228437Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7228830Z E   ModuleNotFoundError: No module named 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7229199Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7229418Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7230091Z _ ERROR at setup of test_enabled_loader_errors_propagate_without_fixture_or_output[error0] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7230584Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7230872Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea4b002f0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7231208Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7231299Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7231572Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7231966Z         """Import a fresh example while proving import-time GPU safety."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7232277Z         blocked = {
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7232463Z             "warp",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7232663Z             "particula.gpu",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7232915Z             "particula.execution.gpu_session",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7233202Z             "particula.execution.gpu_resources",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7233487Z             "particula.execution.checkpoint",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7233737Z         }
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7233942Z         original_import = builtins.__import__
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7234180Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7234350Z         def guarded_import(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7234561Z             name: str,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7234798Z             globals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7235101Z             locals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7235382Z             fromlist: Sequence[str] = (),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7235625Z             level: int = 0,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7235832Z         ) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7236032Z             if name in blocked:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7236416Z                 pytest.fail(f"example imported {name} eagerly")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7236905Z             return original_import(name, globals, locals, fromlist, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7237209Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7237435Z         monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7237761Z         sys.modules.pop("gpu_resident_session", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7238104Z         monkeypatch.setattr(builtins, "__import__", guarded_import)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7238472Z >       module = importlib.import_module("gpu_resident_session")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7238788Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7238967Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7239209Z particula/execution/tests/gpu_resident_session_example_test.py:53: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7239576Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7240574Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7242121Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7242615Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7243056Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7243447Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7243778Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7244082Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7244306Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7244522Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7244619Z name = 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7244886Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7245083Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7245166Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7245428Z E   ModuleNotFoundError: No module named 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7245670Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7245811Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7246355Z _ ERROR at setup of test_enabled_loader_errors_propagate_without_fixture_or_output[error1] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7246685Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7246880Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea4b06180>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7247162Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7247251Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7247529Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7247927Z         """Import a fresh example while proving import-time GPU safety."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7248257Z         blocked = {
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7248456Z             "warp",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7248671Z             "particula.gpu",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7248919Z             "particula.execution.gpu_session",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7249206Z             "particula.execution.gpu_resources",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7249490Z             "particula.execution.checkpoint",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7249741Z         }
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7249941Z         original_import = builtins.__import__
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7250190Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7250369Z         def guarded_import(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7250581Z             name: str,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7250823Z             globals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7251129Z             locals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7251406Z             fromlist: Sequence[str] = (),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7251649Z             level: int = 0,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7251857Z         ) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7252046Z             if name in blocked:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7252309Z                 pytest.fail(f"example imported {name} eagerly")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7252669Z             return original_import(name, globals, locals, fromlist, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7252966Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7253194Z         monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7253514Z         sys.modules.pop("gpu_resident_session", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7253945Z         monkeypatch.setattr(builtins, "__import__", guarded_import)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7254316Z >       module = importlib.import_module("gpu_resident_session")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7254634Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7254813Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7254980Z particula/execution/tests/gpu_resident_session_example_test.py:53: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7255347Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7256306Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7257246Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7257563Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7257856Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7258120Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7258335Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7258595Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7258821Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7259033Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7259132Z name = 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7259394Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7259577Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7259658Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7259898Z E   ModuleNotFoundError: No module named 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7260121Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7260256Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7260625Z ________ ERROR at setup of test_main_propagates_an_enabled_loader_error ________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7260878Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7261058Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea4b00ce0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7261319Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7261405Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7261675Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7262047Z         """Import a fresh example while proving import-time GPU safety."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7262361Z         blocked = {
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7262555Z             "warp",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7262760Z             "particula.gpu",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7263010Z             "particula.execution.gpu_session",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7263302Z             "particula.execution.gpu_resources",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7263591Z             "particula.execution.checkpoint",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7263841Z         }
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7264053Z         original_import = builtins.__import__
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7264305Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7264491Z         def guarded_import(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7264712Z             name: str,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7264961Z             globals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7265296Z             locals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7265588Z             fromlist: Sequence[str] = (),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7265843Z             level: int = 0,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7266059Z         ) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7266344Z             if name in blocked:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7266623Z                 pytest.fail(f"example imported {name} eagerly")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7267173Z             return original_import(name, globals, locals, fromlist, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7267693Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7268064Z         monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7268611Z         sys.modules.pop("gpu_resident_session", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7269285Z         monkeypatch.setattr(builtins, "__import__", guarded_import)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7269912Z >       module = importlib.import_module("gpu_resident_session")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7270547Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7270879Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7271174Z particula/execution/tests/gpu_resident_session_example_test.py:53: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7271946Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7273580Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7275403Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7275916Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7276565Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7277073Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7277401Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7277807Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7278043Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7278267Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7278368Z name = 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7278649Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7278843Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7278925Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7279182Z E   ModuleNotFoundError: No module named 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7279418Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7279558Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7279952Z _____ ERROR at setup of test_build_particle_data_returns_documented_shapes _____
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7280224Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7280409Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0871160>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7280680Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7280774Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7281100Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7281495Z         """Load the published top-level example module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7281820Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7282172Z         sys.modules.pop("data_containers_and_gpu_foundations", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7282488Z >       return _load_module(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7282748Z             "data_containers_and_gpu_foundations_test",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7283023Z             EXAMPLE_PATH,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7283232Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7283342Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7283482Z particula/gpu/tests/data_containers_example_test.py:63: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7283836Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7284243Z particula/gpu/tests/data_containers_example_test.py:54: in _load_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7284613Z     spec.loader.exec_module(module)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7284930Z <frozen importlib._bootstrap_external>:995: in exec_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7285226Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7285471Z <frozen importlib._bootstrap_external>:1132: in get_code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7285764Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7285997Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7286549Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7286751Z self = <_frozen_importlib_external.SourceFileLoader object at 0x7f3ea0871730>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7287210Z path = '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7287461Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7287543Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7287961Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7288539Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7288812Z <frozen importlib._bootstrap_external>:1190: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7289516Z __ ERROR at setup of test_build_gas_data_returns_documented_shapes_and_names ___
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7289937Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7290126Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0871a60>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7290396Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7290479Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7290800Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7291276Z         """Load the published top-level example module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7291618Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7291991Z         sys.modules.pop("data_containers_and_gpu_foundations", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7292327Z >       return _load_module(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7292600Z             "data_containers_and_gpu_foundations_test",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7292891Z             EXAMPLE_PATH,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7293112Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7293226Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7293374Z particula/gpu/tests/data_containers_example_test.py:63: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7293794Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7294186Z particula/gpu/tests/data_containers_example_test.py:54: in _load_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7294542Z     spec.loader.exec_module(module)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7294845Z <frozen importlib._bootstrap_external>:995: in exec_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7295135Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7295377Z <frozen importlib._bootstrap_external>:1132: in get_code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7295664Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7295897Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7296224Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7296425Z self = <_frozen_importlib_external.SourceFileLoader object at 0x7f3ea0871b20>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7297030Z path = '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7297470Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7297596Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7298591Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7299231Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7299454Z <frozen importlib._bootstrap_external>:1190: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7299876Z _____ ERROR at setup of test_warp_enabled_honors_force_no_warp_environment _____
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7300154Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7300340Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0872840>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7300621Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7300707Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7301032Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7301426Z         """Load the published top-level example module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7301762Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7302124Z         sys.modules.pop("data_containers_and_gpu_foundations", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7302449Z >       return _load_module(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7302709Z             "data_containers_and_gpu_foundations_test",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7302995Z             EXAMPLE_PATH,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7303208Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7303316Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7303457Z particula/gpu/tests/data_containers_example_test.py:63: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7303809Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7304197Z particula/gpu/tests/data_containers_example_test.py:54: in _load_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7304555Z     spec.loader.exec_module(module)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7304859Z <frozen importlib._bootstrap_external>:995: in exec_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7305149Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7305383Z <frozen importlib._bootstrap_external>:1132: in get_code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7305660Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7305883Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7306098Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7306376Z self = <_frozen_importlib_external.SourceFileLoader object at 0x7f3ea08729c0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7306829Z path = '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7307078Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7307157Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7307553Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7307936Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7308102Z <frozen importlib._bootstrap_external>:1190: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7308641Z _ ERROR at setup of test_run_example_reports_cpu_only_message_when_warp_disabled _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7308934Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7309123Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0873740>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7309393Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7309477Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7309798Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7310205Z         """Load the published top-level example module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7310541Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7310906Z         sys.modules.pop("data_containers_and_gpu_foundations", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7311445Z >       return _load_module(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7311891Z             "data_containers_and_gpu_foundations_test",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7312372Z             EXAMPLE_PATH,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7312691Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7312850Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7313082Z particula/gpu/tests/data_containers_example_test.py:63: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7313464Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7313867Z particula/gpu/tests/data_containers_example_test.py:54: in _load_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7314221Z     spec.loader.exec_module(module)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7314525Z <frozen importlib._bootstrap_external>:995: in exec_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7314816Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7315050Z <frozen importlib._bootstrap_external>:1132: in get_code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7315331Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7315561Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7315779Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7315973Z self = <_frozen_importlib_external.SourceFileLoader object at 0x7f3ea08738c0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7316512Z path = '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7316770Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7316849Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7317261Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7317642Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7317804Z <frozen importlib._bootstrap_external>:1190: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7318246Z _ ERROR at setup of test_run_example_falls_back_to_cpu_message_when_warp_transfer_fails _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7318552Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7318740Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea08b40e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7319012Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7319094Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7319407Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7319814Z         """Load the published top-level example module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7320144Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7320495Z         sys.modules.pop("data_containers_and_gpu_foundations", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7320810Z >       return _load_module(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7321066Z             "data_containers_and_gpu_foundations_test",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7321349Z             EXAMPLE_PATH,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7321558Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7321666Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7321805Z particula/gpu/tests/data_containers_example_test.py:63: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7322146Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7322539Z particula/gpu/tests/data_containers_example_test.py:54: in _load_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7322897Z     spec.loader.exec_module(module)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7323200Z <frozen importlib._bootstrap_external>:995: in exec_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7323491Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7323731Z <frozen importlib._bootstrap_external>:1132: in get_code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7324015Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7324239Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7324458Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7324648Z self = <_frozen_importlib_external.SourceFileLoader object at 0x7f3ea08b4260>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7325184Z path = '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7325440Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7325521Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7325924Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7326419Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7326592Z <frozen importlib._bootstrap_external>:1190: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7327254Z __________ ERROR at setup of test_example_main_prints_example_output ___________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7327710Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7328021Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea08b4f50>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7328702Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7328831Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7329382Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7330005Z         """Load the published top-level example module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7330554Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7331124Z         sys.modules.pop("data_containers_and_gpu_foundations", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7331625Z >       return _load_module(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7332028Z             "data_containers_and_gpu_foundations_test",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7332441Z             EXAMPLE_PATH,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7332737Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7332887Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7333095Z particula/gpu/tests/data_containers_example_test.py:63: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7333622Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7334257Z particula/gpu/tests/data_containers_example_test.py:54: in _load_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7334625Z     spec.loader.exec_module(module)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7334924Z <frozen importlib._bootstrap_external>:995: in exec_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7335210Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7335581Z <frozen importlib._bootstrap_external>:1132: in get_code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7336052Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7336556Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7336944Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7337354Z self = <_frozen_importlib_external.SourceFileLoader object at 0x7f3ea08b53a0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7338051Z path = '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7338450Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7338559Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7339187Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7339791Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7340038Z <frozen importlib._bootstrap_external>:1190: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7340667Z ___________ ERROR at setup of test_guide_main_prints_example_output ____________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7341078Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7341371Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea08b5e50>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7341790Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7341900Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7342358Z     def guide_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7342741Z         """Load the guide-local forwarding module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7343070Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7343422Z         sys.modules.pop("data_containers_and_gpu_foundations", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7343743Z >       return _load_module(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7344017Z             "data_containers_and_gpu_foundations_guide_test",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7344309Z             GUIDE_PATH,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7344515Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7344626Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7344768Z particula/gpu/tests/data_containers_example_test.py:74: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7345118Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7345514Z particula/gpu/tests/data_containers_example_test.py:54: in _load_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7345879Z     spec.loader.exec_module(module)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7346316Z <frozen importlib._bootstrap_external>:995: in exec_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7346844Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7347151Z <frozen importlib._bootstrap_external>:1132: in get_code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7347435Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7347660Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7347877Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7348067Z self = <_frozen_importlib_external.SourceFileLoader object at 0x7f3ea08b6210>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7348546Z path = '$SRC_DIR/docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7348848Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7348922Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7349378Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7349880Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7350043Z <frozen importlib._bootstrap_external>:1190: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7350465Z _____ ERROR at setup of test_guide_module_re_exports_canonical_run_example _____
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7350731Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7350918Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea08ee4e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7351187Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7351271Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7351581Z     def guide_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7351963Z         """Load the guide-local forwarding module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7352292Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7352649Z         sys.modules.pop("data_containers_and_gpu_foundations", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7352971Z >       return _load_module(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7353247Z             "data_containers_and_gpu_foundations_guide_test",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7353538Z             GUIDE_PATH,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7353742Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7353851Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7353991Z particula/gpu/tests/data_containers_example_test.py:74: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7354329Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7354718Z particula/gpu/tests/data_containers_example_test.py:54: in _load_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7355077Z     spec.loader.exec_module(module)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7355375Z <frozen importlib._bootstrap_external>:995: in exec_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7355664Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7355892Z <frozen importlib._bootstrap_external>:1132: in get_code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7356369Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7356727Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7357087Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7357376Z self = <_frozen_importlib_external.SourceFileLoader object at 0x7f3ea08ed9d0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7358138Z path = '$SRC_DIR/docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7358615Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7358710Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7359410Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7360045Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7360295Z <frozen importlib._bootstrap_external>:1190: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7361014Z _ ERROR at setup of test_guide_module_raises_import_error_when_canonical_example_cannot_load _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7361517Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7361801Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea08b75f0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7362224Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7362336Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7362791Z     def guide_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7363377Z         """Load the guide-local forwarding module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7363714Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7364072Z         sys.modules.pop("data_containers_and_gpu_foundations", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7364395Z >       return _load_module(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7364668Z             "data_containers_and_gpu_foundations_guide_test",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7364962Z             GUIDE_PATH,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7365165Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7365347Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7365491Z particula/gpu/tests/data_containers_example_test.py:74: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7365837Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7366317Z particula/gpu/tests/data_containers_example_test.py:54: in _load_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7366777Z     spec.loader.exec_module(module)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7367258Z <frozen importlib._bootstrap_external>:995: in exec_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7367693Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7368028Z <frozen importlib._bootstrap_external>:1132: in get_code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7368609Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7369000Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7369339Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7369631Z self = <_frozen_importlib_external.SourceFileLoader object at 0x7f3ea08b76b0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7370380Z path = '$SRC_DIR/docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7370864Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7370962Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7371557Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7371982Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7372146Z <frozen importlib._bootstrap_external>:1190: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7372577Z _ ERROR at setup of test_run_example_warp_path_reports_round_trip_shapes_and_names _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7372853Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7373054Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0ac7680>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7373334Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7373421Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7373741Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7374142Z         """Load the published top-level example module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7374485Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7374859Z         sys.modules.pop("data_containers_and_gpu_foundations", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7375166Z >       return _load_module(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7375416Z             "data_containers_and_gpu_foundations_test",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7375690Z             EXAMPLE_PATH,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7375894Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7375998Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7376237Z particula/gpu/tests/data_containers_example_test.py:63: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7376584Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7376960Z particula/gpu/tests/data_containers_example_test.py:54: in _load_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7377315Z     spec.loader.exec_module(module)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7377610Z <frozen importlib._bootstrap_external>:995: in exec_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7377899Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7378129Z <frozen importlib._bootstrap_external>:1132: in get_code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7378401Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7378619Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7378827Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7379010Z self = <_frozen_importlib_external.SourceFileLoader object at 0x7f3ea0872ba0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7379437Z path = '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7379679Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7379752Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7380149Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7380518Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7380674Z <frozen importlib._bootstrap_external>:1190: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7381071Z _ ERROR at setup of test_cpu_fixture_has_documented_active_and_inactive_slots __
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7381329Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7381648Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0872ae0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7382064Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7382174Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7382626Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7383343Z         """Load the standalone example without retained module state."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7383886Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7384502Z         monkeypatch.delitem(sys.modules, "gpu_coagulation_direct", raising=False)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7385142Z >       return importlib.import_module("gpu_coagulation_direct")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7385622Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7385890Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7386226Z particula/gpu/tests/gpu_coagulation_direct_example_test.py:39: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7386783Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7388271Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7389729Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7390217Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7390645Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7391035Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7391345Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7391740Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7392185Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7392515Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7392645Z name = 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7393022Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7393324Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7393418Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7393797Z E   ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7394169Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7394372Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7394959Z _____ ERROR at setup of test_runtime_loader_uses_selected_adapter_imports ______
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7395368Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7395657Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea08ee930>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7396062Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7396257Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7396723Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7397348Z         """Load the standalone example without retained module state."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7397884Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7398486Z         monkeypatch.delitem(sys.modules, "gpu_coagulation_direct", raising=False)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7399127Z >       return importlib.import_module("gpu_coagulation_direct")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7399612Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7399880Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7400124Z particula/gpu/tests/gpu_coagulation_direct_example_test.py:39: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7400670Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7401754Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7402661Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7402989Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7403307Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7403597Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7403830Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7404102Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7404338Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7404566Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7404661Z name = 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7404928Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7405189Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7405367Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7405647Z E   ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7405910Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7406055Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7406729Z _ ERROR at setup of test_enabled_path_uses_selected_adapter_and_explicit_lifecycle _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7407230Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7407556Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea08ef650>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7408016Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7408145Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7408644Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7409416Z         """Load the standalone example without retained module state."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7409997Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7410617Z         monkeypatch.delitem(sys.modules, "gpu_coagulation_direct", raising=False)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7411276Z >       return importlib.import_module("gpu_coagulation_direct")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7411765Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7412045Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7412297Z particula/gpu/tests/gpu_coagulation_direct_example_test.py:39: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7412864Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7414494Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7415989Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7416580Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7417033Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7417419Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7417742Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7418150Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7418471Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7418816Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7418951Z name = 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7419336Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7419740Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7419841Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7420226Z E   ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7420595Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7420804Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7421420Z _ ERROR at setup of test_failures_propagate_without_fallback_or_restore[loader] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7421860Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7422146Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea2686060>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7422579Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7422692Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7423166Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7423827Z         """Load the standalone example without retained module state."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7424321Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7424713Z         monkeypatch.delitem(sys.modules, "gpu_coagulation_direct", raising=False)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7425133Z >       return importlib.import_module("gpu_coagulation_direct")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7425461Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7425639Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7425804Z particula/gpu/tests/gpu_coagulation_direct_example_test.py:39: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7426275Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7427168Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7428145Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7428473Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7428771Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7429040Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7429267Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7429541Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7429776Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7429996Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7430095Z name = 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7430433Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7430634Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7430716Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7430973Z E   ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7431209Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7431354Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7432017Z _ ERROR at setup of test_failures_propagate_without_fallback_or_restore[conversion] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7432478Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7432763Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea08b5cd0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7433198Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7433312Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7433769Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7434415Z         """Load the standalone example without retained module state."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7434970Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7435573Z         monkeypatch.delitem(sys.modules, "gpu_coagulation_direct", raising=False)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7436327Z >       return importlib.import_module("gpu_coagulation_direct")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7436833Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7437111Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7437352Z particula/gpu/tests/gpu_coagulation_direct_example_test.py:39: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7437916Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7439337Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7440788Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7441270Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7441713Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7442112Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7442443Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7442850Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7443174Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7443614Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7443740Z name = 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7444149Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7444462Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7444573Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7444960Z E   ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7445352Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7445572Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7446295Z _ ERROR at setup of test_failures_propagate_without_fallback_or_restore[first] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7446741Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7447038Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea08ee180>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7447472Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7447591Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7448070Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7448748Z         """Load the standalone example without retained module state."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7449328Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7449963Z         monkeypatch.delitem(sys.modules, "gpu_coagulation_direct", raising=False)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7450726Z >       return importlib.import_module("gpu_coagulation_direct")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7451230Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7451521Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7451766Z particula/gpu/tests/gpu_coagulation_direct_example_test.py:39: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7452331Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7453833Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7455070Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7455424Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7455733Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7456015Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7456382Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7456665Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7456894Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7457115Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7457212Z name = 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7457494Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7457698Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7457782Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7458050Z E   ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7458292Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7458441Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7458913Z _ ERROR at setup of test_failures_propagate_without_fallback_or_restore[second] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7459407Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7459705Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea08ef890>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7460144Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7460260Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7460743Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7461425Z         """Load the standalone example without retained module state."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7462024Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7462658Z         monkeypatch.delitem(sys.modules, "gpu_coagulation_direct", raising=False)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7463326Z >       return importlib.import_module("gpu_coagulation_direct")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7463827Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7464119Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7464369Z particula/gpu/tests/gpu_coagulation_direct_example_test.py:39: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7465062Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7466631Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7468145Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7468642Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7469179Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7469597Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7469928Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7470342Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7470673Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7471017Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7471156Z name = 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7471527Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7471731Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7471812Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7472076Z E   ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7472319Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7472469Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7472968Z _ ERROR at setup of test_failures_propagate_without_fallback_or_restore[synchronize] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7473288Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7473480Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07784a0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7473759Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7473844Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7474168Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7474604Z         """Load the standalone example without retained module state."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7475006Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7475480Z         monkeypatch.delitem(sys.modules, "gpu_coagulation_direct", raising=False)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7475913Z >       return importlib.import_module("gpu_coagulation_direct")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7476362Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7476556Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7476723Z particula/gpu/tests/gpu_coagulation_direct_example_test.py:39: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7477095Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7478018Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7478944Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7479281Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7479591Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7479867Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7480100Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7480376Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7480607Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7480837Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7480944Z name = 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7481224Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7481427Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7481561Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7482002Z E   ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7482404Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7482621Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7483293Z _ ERROR at setup of test_failures_propagate_without_fallback_or_restore[restore] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7483749Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7484056Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0779280>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7484510Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7484618Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7485084Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7485741Z         """Load the standalone example without retained module state."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7486385Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7487008Z         monkeypatch.delitem(sys.modules, "gpu_coagulation_direct", raising=False)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7487652Z >       return importlib.import_module("gpu_coagulation_direct")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7488147Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7488422Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7488661Z particula/gpu/tests/gpu_coagulation_direct_example_test.py:39: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7489211Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7490752Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7492226Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7492713Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7493161Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7493653Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7493996Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7494393Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7494722Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7495056Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7495191Z name = 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7495579Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7495878Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7495985Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7496464Z E   ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7496983Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7497196Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7497837Z _ ERROR at setup of test_real_warp_selected_adapter_reuses_rng_and_preserves_identity _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7498303Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7498584Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea08ee6f0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7499010Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7499130Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7499600Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7500262Z         """Load the standalone example without retained module state."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7500824Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7501428Z         monkeypatch.delitem(sys.modules, "gpu_coagulation_direct", raising=False)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7502073Z >       return importlib.import_module("gpu_coagulation_direct")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7502556Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7502839Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7503068Z particula/gpu/tests/gpu_coagulation_direct_example_test.py:39: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7503624Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7505069Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7507312Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7507812Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7508255Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7508660Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7508978Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7509382Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7509715Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7510053Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7510185Z name = 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7510582Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7510893Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7511000Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7511380Z E   ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7511743Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7511953Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7512555Z _ ERROR at setup of test_build_cpu_state_has_documented_sparse_float64_schema __
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7512974Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7513258Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea08b6540>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7513674Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7513782Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7514185Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7514758Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7515363Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7515918Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7516799Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7517776Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7518656Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7518971Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7519239Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7519829Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7521279Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7522725Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7523322Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7523774Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7524169Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7524491Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7524887Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7525215Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7525548Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7525694Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7526208Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7526509Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7526619Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7527028Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7527441Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7527649Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7528248Z __ ERROR at setup of test_forced_disabled_path_does_not_reach_enabled_loader ___
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7528674Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7528962Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0779760>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7529380Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7529499Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7529928Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7530509Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7531121Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7531678Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7532189Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7532786Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7533330Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7533640Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7533913Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7534524Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7535990Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7537578Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7538072Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7538510Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7538916Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7539252Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7539665Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7539995Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7540334Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7540493Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7540917Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7541213Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7541324Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7541739Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7542172Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7542374Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7543083Z _ ERROR at setup of test_warp_enabled_handles_import_failure_and_available_runtime _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7543542Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7543820Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea077bbc0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7544242Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7544364Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7544772Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7545347Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7545937Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7546820Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7547908Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7548522Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7549082Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7549389Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7549688Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7549887Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7551048Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7551275Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7551431Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7551604Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7551713Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7551904Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7551996Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7552194Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7552200Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7552348Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7552524Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7552530Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7552630Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7552902Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7552908Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7553117Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7553438Z _ ERROR at setup of test_runtime_unavailable_returns_metadata_without_device_transfers _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7553444Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7553731Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07bc7a0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7553741Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7553863Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7554086Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7554300Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7554549Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7554815Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7555107Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7555367Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7555918Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7555929Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7556287Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7556858Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7559155Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7559475Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7560460Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7560845Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7560966Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7561561Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7561682Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7562177Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7562288Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7562718Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7562899Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7562986Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7563341Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7563641Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7563648Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7564024Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7564837Z _ ERROR at setup of test_enabled_runtime_loading_failure_propagates_without_success_output _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7564848Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7565382Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0779670>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7565393Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7565689Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7566364Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7566869Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7567322Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7567799Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7568123Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7568757Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7569289Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7569301Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7569677Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7570198Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7572489Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7572916Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7573276Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7573957Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7574086Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7574279Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7574381Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7574877Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7574888Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7575050Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7575229Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7575235Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7575344Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7575724Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7575733Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7575952Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7576730Z _ ERROR at setup of test_setup_conversion_failure_does_not_continue_to_sidecars_or_steps[to_warp_particle_data] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7576741Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7577479Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0778da0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7577496Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7577627Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7578161Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7578403Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7578662Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7579026Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7579250Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7579608Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7579780Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7579788Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7580058Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7580341Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7581680Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7582093Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7582253Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7582428Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7582539Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7582732Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7582943Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7583151Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7583158Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7583301Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7583473Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7583479Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7583681Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7583986Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7583993Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7584199Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7584741Z _ ERROR at setup of test_setup_conversion_failure_does_not_continue_to_sidecars_or_steps[to_warp_gas_data] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7584752Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7585046Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07bcb90>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7585051Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7585273Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7585513Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7585730Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7586218Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7586496Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7586702Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7587165Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7587320Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7587326Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7587593Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7587896Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7589175Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7589403Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7589663Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7589830Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7589938Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7590116Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7590312Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7590585Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7590595Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7590743Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7590918Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7591004Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7591102Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7591487Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7591496Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7591692Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7592318Z _ ERROR at setup of test_setup_conversion_failure_does_not_continue_to_sidecars_or_steps[to_warp_environment_data] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7592329Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7592612Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07bd730>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7592619Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7592805Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7593147Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7593364Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7593608Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7593947Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7594169Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7594427Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7594682Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7594690Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7594968Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7595148Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7596578Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7596803Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7596943Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7597233Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7597336Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7597521Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7597618Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7597805Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7597811Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7598068Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7598234Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7598241Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7598360Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7598727Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7598743Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7598946Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7599267Z _ ERROR at setup of test_load_enabled_runtime_defers_and_collects_required_boundaries _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7599273Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7599657Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07be780>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7599672Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7599790Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7600008Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7600317Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7600676Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7600844Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7601060Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7601424Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7601604Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7601611Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7601878Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7602170Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7603584Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7604100Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7604259Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7604914Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7605032Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7605224Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7605751Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7606036Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7606043Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7606319Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7606923Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7606933Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7607029Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7607503Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7607516Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7607735Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7608408Z _ ERROR at setup of test_enabled_path_converts_once_orders_steps_and_restores_once _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7608418Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7608712Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07bf980>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7608718Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7608832Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7609072Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7609304Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7609555Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7609733Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7609942Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7610212Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7610359Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7610374Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7610619Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7610752Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7611422Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7611579Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7611685Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7611800Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7611884Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7612008Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7612089Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7612216Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7612220Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7612328Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7612448Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7612452Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7612526Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7612710Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7612714Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7612853Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7613151Z _ ERROR at setup of test_boundary_failure_propagates_stops_later_calls_and_prevents_early_restore[condensation] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7613155Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7613346Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3eae2c0c80>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7613350Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7613497Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7613656Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7613802Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7613973Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7614089Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7614236Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7614409Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7614576Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7614581Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7614764Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7614890Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7615559Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7615711Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7615812Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7615934Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7616008Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7616244Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7616325Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7616629Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7616639Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7616891Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7617088Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7617095Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7617222Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7617636Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7617645Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7617854Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7618430Z _ ERROR at setup of test_boundary_failure_propagates_stops_later_calls_and_prevents_early_restore[coagulation] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7618441Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7618724Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07bfe00>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7618730Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7618855Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7619080Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7619310Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7619552Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7619729Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7619955Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7620128Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7620244Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7620248Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7620422Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7620559Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7621234Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7621385Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7621492Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7621607Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7621790Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7622261Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7622352Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7622493Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7622497Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7622721Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7622887Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7622892Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7622965Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7623149Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7623153Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7623298Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7623792Z _ ERROR at setup of test_boundary_failure_propagates_stops_later_calls_and_prevents_early_restore[dilution] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7623800Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7624043Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07bc950>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7624053Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7624159Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7624448Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7624677Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7624924Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7625102Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7625309Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7625571Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7625724Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7625743Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7626012Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7626301Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7627448Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7627676Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7627816Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7628001Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7628108Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7628235Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7628317Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7628446Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7628453Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7628564Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7628679Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7628690Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7628765Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7628951Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7628958Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7629092Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7629381Z _ ERROR at setup of test_boundary_failure_propagates_stops_later_calls_and_prevents_early_restore[wall_loss] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7629385Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7629569Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0772c60>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7629579Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7629661Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7629819Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7629963Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7630138Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7630252Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7630398Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7630572Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7630742Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7630747Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7630934Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7631061Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7631728Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7631964Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7632066Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7632192Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7632268Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7632398Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7632476Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7632610Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7632614Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7632724Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7632837Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7632841Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7632920Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7633096Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7633100Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7633243Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7633528Z _ ERROR at setup of test_boundary_failure_propagates_stops_later_calls_and_prevents_early_restore[nucleation] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7633532Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7633714Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0770ce0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7633718Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7633806Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7633957Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7634111Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7634274Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7634400Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7634545Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7634708Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7634822Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7634828Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7635004Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7635138Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7635808Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7635953Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7636060Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7636280Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7636371Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7636503Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7636576Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7636709Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7636718Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7636820Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7636942Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7636946Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7637020Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7637206Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7637210Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7637567Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7637853Z _ ERROR at setup of test_boundary_failure_propagates_stops_later_calls_and_prevents_early_restore[synchronize] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7637858Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7638049Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0771be0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7638053Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7638135Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7638296Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7638446Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7638677Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7638804Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7638942Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7639114Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7639226Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7639239Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7639414Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7639547Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7640468Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7641796Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7642227Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7642598Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7643017Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7643294Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7643685Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7644021Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7644369Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7644526Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7644997Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7645322Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7645440Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7645885Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7646481Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7646724Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7647599Z _ ERROR at setup of test_boundary_failure_propagates_stops_later_calls_and_prevents_early_restore[restore_particles] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7648288Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7648605Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07791f0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7649074Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7649251Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7649678Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7650230Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7650719Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7651093Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7651435Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7651823Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7652184Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7652391Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7652582Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7652970Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7654004Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7654925Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7655266Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7655573Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7655847Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7656080Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7656504Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7656737Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7657034Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7657145Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7657437Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7657633Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7657717Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7657995Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7658259Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7658399Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7658891Z _ ERROR at setup of test_boundary_failure_propagates_stops_later_calls_and_prevents_early_restore[restore_gas] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7659271Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7659467Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea08b4140>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7659729Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7659821Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7660097Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7660473Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7660863Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7661230Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7661564Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7661947Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7662313Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7662505Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7662692Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7663081Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7663949Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7665012Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7665433Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7665848Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7666329Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7666701Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7667071Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7667309Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7667532Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7667634Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7667921Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7668279Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7668389Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7668904Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7669350Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7669557Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7670184Z _ ERROR at setup of test_boundary_failure_propagates_stops_later_calls_and_prevents_early_restore[restore_environment] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7670571Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7670766Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0771790>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7671033Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7671119Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7671394Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7671841Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7672233Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7672604Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7672933Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7673317Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7673673Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7673873Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7674052Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7674508Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7675396Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7676408Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7676741Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7677045Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7677314Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7677540Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7677813Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7678038Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7678261Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7678367Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7678660Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7678855Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7678939Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7679215Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7679473Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7679617Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7680001Z ____________ ERROR at setup of test_main_prints_only_example_output ____________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7680264Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7680445Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0770b90>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7680714Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7680797Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7681069Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7681438Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7681829Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7682194Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7682525Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7682908Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7683261Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7683458Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7683639Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7684025Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7684911Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7685811Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7686214Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7686536Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7686976Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7687222Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7687492Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7687715Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7688027Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7688130Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7688526Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7688720Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7688804Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7689081Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7689337Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7689484Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7689876Z ___ ERROR at setup of test_real_warp_cpu_path_restores_named_cpu_containers ____
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7690149Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7690397Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07736e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7690664Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7690745Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7691123Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7691510Z         """Import a fresh example module without requiring Warp."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7691996Z         examples = Path(__file__).resolve().parents[3] / "docs" / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7692537Z         monkeypatch.syspath_prepend(str(examples))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7693038Z         sys.modules.pop("gpu_complete_process_sequence", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7693617Z >       module = importlib.import_module("gpu_complete_process_sequence")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7694207Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7694551Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7694824Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:35: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7695420Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7696458Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7697396Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7697738Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7698043Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7698315Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7698547Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7698830Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7699058Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7699282Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7699393Z name = 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7699679Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7699876Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7699959Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7700238Z E   ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7700502Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7700640Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7701052Z _ ERROR at setup of test_fixture_and_builders_are_fp64_readonly_and_non_aliasing _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7701342Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7701529Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07e8590>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7701802Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7701893Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7702212Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7702657Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7703029Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7703417Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7703846Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7704232Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7704442Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7704620Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7705082Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7705958Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7706926Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7707264Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7707570Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7707842Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7708140Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7708414Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7708645Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7708863Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7708983Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7709287Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7709483Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7709602Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7710058Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7710456Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7710601Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7711071Z _ ERROR at setup of test_oracle_has_four_substep_uptake_evaporation_coupling_and_energy _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7711611Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7711912Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0771f70>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7712316Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7712434Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7712805Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7713215Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7713582Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7713963Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7714382Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7714764Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7714966Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7715142Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7715527Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7716489Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7717395Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7717725Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7718032Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7718301Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7718528Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7718803Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7719040Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7719256Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7719382Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7719684Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7719872Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7719954Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7720246Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7720521Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7720662Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7721046Z ________ ERROR at setup of test_oracle_does_not_mutate_supplied_source _________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7721307Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7721490Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07bdbb0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7721760Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7721849Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7722228Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7722639Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7723005Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7723384Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7723801Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7724179Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7724493Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7724671Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7725059Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7725928Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7726897Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7727235Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7727547Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7727819Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7728050Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7728329Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7728569Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7728788Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7728911Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7729220Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7729410Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7729495Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7729848Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7730283Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7730452Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7731347Z _ ERROR at setup of test_fixture_validation_rejects_nonfinite_and_invalid_physical_values[temperature-value0-finite] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7731927Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7732123Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea08b4500>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7732385Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7732478Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7732788Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7733194Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7733773Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7734451Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7735152Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7735732Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7736039Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7736387Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7736992Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7738198Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7739408Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7739915Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7740377Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7740757Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7741089Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7741491Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7741851Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7742170Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7742344Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7742664Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7742857Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7742940Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7743236Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7743516Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7743652Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7744167Z _ ERROR at setup of test_fixture_validation_rejects_nonfinite_and_invalid_physical_values[pressure-value1-finite] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7744620Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7744812Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07e8d70>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7745076Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7745165Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7745486Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7745878Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7746329Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7746706Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7747112Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7747481Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7747674Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7747853Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7748225Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7749074Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7749942Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7750261Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7750552Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7750812Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7751033Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7751301Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7751522Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7751727Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7751841Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7752134Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7752327Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7752398Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7752680Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7752954Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7753088Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7753587Z _ ERROR at setup of test_fixture_validation_rejects_nonfinite_and_invalid_physical_values[density-value2-positive] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7753960Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7754148Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07ea900>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7754407Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7754497Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7754796Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7755185Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7755535Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7755901Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7756507Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7756979Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7757288Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7757544Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7758030Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7759361Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7760687Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7761170Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7761601Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7761962Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7762316Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7762711Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7763023Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7763336Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7763498Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7763979Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7764272Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7764380Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7764816Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7765268Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7765457Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7766421Z _ ERROR at setup of test_fixture_validation_rejects_nonfinite_and_invalid_physical_values[gas_concentration-value3-nonnegative] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7767102Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7767409Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07eb920>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7767813Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7767935Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7768406Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7769073Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7769599Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7770168Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7770818Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7771393Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7771690Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7771953Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7772517Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7773948Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7775382Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7775889Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7776461Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7776914Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7777245Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7777688Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7778016Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7778374Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7778551Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7779021Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7779329Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7779442Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7779914Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7780368Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7780603Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7781238Z _ ERROR at setup of test_fixture_validation_rejects_unsupported_thermodynamic_mode _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7781719Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7782023Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07bec30>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7782564Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7782684Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7783153Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7783818Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7784390Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7784957Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7785661Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7786368Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7786750Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7787045Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7787634Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7789156Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7790684Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7791188Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7791687Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7792113Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7792432Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7792832Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7793153Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7793487Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7793673Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7794177Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7794515Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7794647Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7795112Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7795561Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7795792Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7796548Z _ ERROR at setup of test_disabled_or_unavailable_warp_completes_oracle_without_runtime_work _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7797106Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7797415Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07eb350>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7797873Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7798003Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7798508Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7799100Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7799536Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7800088Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7800813Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7801455Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7801732Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7802043Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7802584Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7803455Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7804332Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7804664Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7804976Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7805245Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7805473Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7805746Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7806053Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7806396Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7806514Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7806818Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7807017Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7807092Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7807389Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7807674Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7807822Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7808221Z ____ ERROR at setup of test_force_disabled_warp_defers_runtime_after_oracle ____
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7808556Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7808752Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07ea240>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7809023Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7809118Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7809431Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7809842Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7810203Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7810585Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7811015Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7811395Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7811593Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7811775Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7812161Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7813616Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7814785Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7815118Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7815425Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7815699Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7815929Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7816499Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7816880Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7817251Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7817434Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7817946Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7818303Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7818402Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7818700Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7818998Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7819144Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7819581Z _ ERROR at setup of test_fake_enabled_route_has_explicit_sidecars_and_synchronized_readback _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7819889Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7820082Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07c0260>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7820357Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7820442Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7820756Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7821164Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7821528Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7821916Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7822342Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7822719Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7822912Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7823095Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7823550Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7824416Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7825305Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7825638Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7825937Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7826431Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7826664Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7826937Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7827164Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7827378Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7827491Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7827798Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7827995Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7828069Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7828371Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7828640Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7828785Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7829215Z _ ERROR at setup of test_oracle_completes_before_runtime_and_ignores_warp_source_mutation _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7829518Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7829713Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07c0fb0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7829985Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7830068Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7830381Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7830782Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7831143Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7831526Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7831943Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7832315Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7832507Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7832686Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7833059Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7833930Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7834819Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7835149Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7835451Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7835717Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7835945Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7836472Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7836829Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7837207Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7837404Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7837878Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7838079Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7838155Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7838516Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7839036Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7839278Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7840068Z _ ERROR at setup of test_enabled_failures_propagate_after_completed_oracle_without_restore[loader] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7840694Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7841048Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07c1e20>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7841641Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7841778Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7842311Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7843064Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7843680Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7844076Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7844497Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7844885Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7845200Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7845379Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7845747Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7846708Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7847583Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7847901Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7848194Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7848458Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7848682Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7848956Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7849187Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7849401Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7849512Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7849808Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7850002Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7850083Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7850374Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7850641Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7850780Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7851266Z _ ERROR at setup of test_enabled_failures_propagate_after_completed_oracle_without_restore[particle conversion] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7851631Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7851812Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07e8920>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7852078Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7852161Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7852463Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7852859Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7853214Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7853581Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7853986Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7854355Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7854547Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7854716Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7855085Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7855937Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7856887Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7857212Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7857506Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7857771Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7857991Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7858260Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7858552Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7858764Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7858874Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7859199Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7859542Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7859660Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7860105Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7860432Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7860673Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7861591Z _ ERROR at setup of test_enabled_failures_propagate_after_completed_oracle_without_restore[gas conversion] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7862286Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7862616Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0772e70>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7863133Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7863248Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7863803Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7864557Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7865209Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7865838Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7866719Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7867376Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7867633Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7867950Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7868672Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7870449Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7871766Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7872113Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7872408Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7872674Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7872898Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7873169Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7873394Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7873607Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7873720Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7874025Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7874220Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7874298Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7874586Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7874856Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7875001Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7875465Z _ ERROR at setup of test_enabled_failures_propagate_after_completed_oracle_without_restore[allocation] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7875809Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7875995Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07c1280>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7876367Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7876454Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7876764Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7876908Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7877057Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7877217Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7877403Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7877513Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7877517Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7877786Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7877922Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7878554Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7878707Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7878805Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7879063Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7879136Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7879265Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7879344Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7879473Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7879477Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7879598Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7879711Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7879715Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7879796Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7879994Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7879998Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7880130Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7880379Z _ ERROR at setup of test_enabled_failures_propagate_after_completed_oracle_without_restore[kernel] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7880383Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7880560Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07c2180>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7880567Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7880657Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7880846Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7880979Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7881128Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7881280Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7881459Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7881565Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7881569Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7881743Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7881872Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7882522Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7882673Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7882768Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7882889Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7882968Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7883090Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7883173Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7883402Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7883411Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7883599Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7883789Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7883803Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7883909Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7884239Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7884246Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7884475Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7884840Z ___ ERROR at setup of test_acceptance_categories_are_independently_evaluated ___
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7884850Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7885241Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07c3770>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7885247Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7885385Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7885651Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7885893Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7886257Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7886524Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7886846Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7887117Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7887124Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7887410Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7887635Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7888733Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7888891Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7888997Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7889111Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7889192Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7889313Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7889394Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7889517Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7889521Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7889638Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7889757Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7889760Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7889834Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7890032Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7890035Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7890165Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7890372Z _ ERROR at setup of test_acceptance_reports_all_categories_after_multiple_failures _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7890376Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7890558Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07b44a0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7890561Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7890640Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7890891Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7891060Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7891259Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7891458Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7891750Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7891892Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7891896Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7892141Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7892332Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7892971Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7893127Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7893235Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7893350Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7893432Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7893551Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7893713Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7893841Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7893845Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7893966Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7894087Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7894091Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7894162Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7894357Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7894362Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7894492Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7894735Z __________ ERROR at setup of test_warp_cpu_matches_independent_oracle __________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7894739Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7894920Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07c17c0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7894924Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7895004Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7895189Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7895319Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7895467Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7895626Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7895798Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7895910Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7895914Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7896076Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7896321Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7896976Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7897123Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7897226Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7897342Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7897424Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7897544Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7897628Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7897758Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7897763Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7897871Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7897994Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7897997Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7898071Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7898267Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7898271Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7898402Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7898590Z ____ ERROR at setup of test_cuda_matches_independent_oracle_when_available _____
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7898594Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7898778Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0771ee0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7898782Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7898862Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7899050Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7899177Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7899327Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7899485Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7899653Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7899768Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7899772Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7899936Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7900129Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7900781Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7900923Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7901029Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7901143Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7901276Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7901397Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7901477Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7901607Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7901610Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7901718Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7901839Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7901842Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7901913Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7902108Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7902112Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7902246Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7902417Z ______ ERROR at setup of test_main_returns_nonzero_for_failed_acceptance _______
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7902421Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7902600Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07b4aa0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7902607Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7902686Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7902870Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7903004Z         """Import the CPU-safe published walkthrough module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7903143Z         monkeypatch.syspath_prepend(str(EXAMPLE_PATH.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7903298Z         sys.modules.pop("gpu_condensation_parity_walkthrough", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7903470Z >       return importlib.import_module("gpu_condensation_parity_walkthrough")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7903583Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7903587Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7903750Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:36: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7903877Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7904520Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7904663Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7904768Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7904885Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7905022Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7905224Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7905327Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7905518Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7905524Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7905679Z name = 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7905869Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7905875Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7905995Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7906414Z E   ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7906429Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7906676Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7906994Z _ ERROR at setup of test_cpu_builders_preserve_documented_dtype_and_species_order _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7906999Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7907329Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07b56d0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7907421Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7907560Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7907907Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7908122Z         """Load the published top-level example module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7908346Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7908595Z         sys.modules.pop("gpu_direct_kernels_quick_start", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7908888Z >       return importlib.import_module("gpu_direct_kernels_quick_start")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7909090Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7909175Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7909379Z particula/gpu/tests/gpu_direct_kernels_example_test.py:59: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7909550Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7910803Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7911053Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7911175Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7911396Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7911518Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7911737Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7911847Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7912037Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7912053Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7912185Z name = 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7912308Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7912312Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7912391Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7912567Z E   ModuleNotFoundError: No module named 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7912573Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7912710Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7912910Z _ ERROR at setup of test_load_gpu_runtime_imports_only_direct_condensation_contract _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7912914Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7913096Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07b6e70>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7913100Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7913187Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7913366Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7913498Z         """Load the published top-level example module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7913626Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7913763Z         sys.modules.pop("gpu_direct_kernels_quick_start", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7913924Z >       return importlib.import_module("gpu_direct_kernels_quick_start")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7914035Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7914038Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7914196Z particula/gpu/tests/gpu_direct_kernels_example_test.py:59: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7914319Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7914959Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7915099Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7915205Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7915329Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7915401Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7915528Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7915602Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7915736Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7915795Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7915896Z name = 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7916017Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7916020Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7916100Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7916381Z E   ModuleNotFoundError: No module named 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7916387Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7916524Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7916702Z _____ ERROR at setup of test_unavailable_warp_skips_loader_and_conversions _____
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7916705Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7916952Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07b6f00>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7916955Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7917044Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7917226Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7917359Z         """Load the published top-level example module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7917485Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7917632Z         sys.modules.pop("gpu_direct_kernels_quick_start", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7917793Z >       return importlib.import_module("gpu_direct_kernels_quick_start")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7917903Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7917907Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7918061Z particula/gpu/tests/gpu_direct_kernels_example_test.py:59: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7918183Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7918829Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7918973Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7919077Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7919201Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7919272Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7919397Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7919469Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7919598Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7919602Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7919697Z name = 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7919815Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7919818Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7919898Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7920078Z E   ModuleNotFoundError: No module named 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7920081Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7920215Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7920390Z __ ERROR at setup of test_enabled_path_reuses_complete_caller_owned_sidecars ___
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7920394Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7920576Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0789fa0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7920579Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7920665Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7920841Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7920973Z         """Load the published top-level example module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7921095Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7921236Z         sys.modules.pop("gpu_direct_kernels_quick_start", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7921400Z >       return importlib.import_module("gpu_direct_kernels_quick_start")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7921500Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7921507Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7921658Z particula/gpu/tests/gpu_direct_kernels_example_test.py:59: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7921779Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7922478Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7922629Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7922726Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7922846Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7922919Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7923046Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7923118Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7923294Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7923298Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7923401Z name = 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7923512Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7923516Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7923592Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7923770Z E   ModuleNotFoundError: No module named 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7923774Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7923909Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7924119Z _ ERROR at setup of test_kernel_failure_propagates_without_restore_or_success_output[1] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7924130Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7924301Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0789640>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7924305Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7924390Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7924569Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7924698Z         """Load the published top-level example module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7924824Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7924963Z         sys.modules.pop("gpu_direct_kernels_quick_start", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7925127Z >       return importlib.import_module("gpu_direct_kernels_quick_start")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7925230Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7925233Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7925382Z particula/gpu/tests/gpu_direct_kernels_example_test.py:59: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7925501Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7926235Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7926392Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7926493Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7926610Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7926680Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7926804Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7926876Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7927008Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7927012Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7927115Z name = 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7927228Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7927232Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7927310Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7927538Z E   ModuleNotFoundError: No module named 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7927547Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7927777Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7928140Z _ ERROR at setup of test_kernel_failure_propagates_without_restore_or_success_output[2] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7928152Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7928472Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea078a660>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7928478Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7928610Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7928838Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7929020Z         """Load the published top-level example module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7929212Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7929347Z         sys.modules.pop("gpu_direct_kernels_quick_start", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7929641Z >       return importlib.import_module("gpu_direct_kernels_quick_start")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7929811Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7929817Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7930072Z particula/gpu/tests/gpu_direct_kernels_example_test.py:59: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7930279Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7931461Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7931726Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7931883Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7932088Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7932202Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7932409Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7932525Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7932724Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7932730Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7932889Z name = 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7933004Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7933008Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7933090Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7933268Z E   ModuleNotFoundError: No module named 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7933272Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7933410Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7933592Z __ ERROR at setup of test_real_warp_cpu_path_reuses_sidecars_and_couples_gas ___
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7933596Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7933775Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea078b5c0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7933778Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7933865Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7934048Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7934177Z         """Load the published top-level example module."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7934306Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7934439Z         sys.modules.pop("gpu_direct_kernels_quick_start", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7934603Z >       return importlib.import_module("gpu_direct_kernels_quick_start")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7934709Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7934712Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7934861Z particula/gpu/tests/gpu_direct_kernels_example_test.py:59: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7934984Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7935635Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7935783Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7935879Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7935999Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7936070Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7936300Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7936390Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7936513Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7936516Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7936619Z name = 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7936732Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7936735Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7936816Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7937102Z E   ModuleNotFoundError: No module named 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7937114Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7937246Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7937420Z __________ ERROR at setup of test_forced_disable_runs_no_enabled_work __________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7937423Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7937594Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0462ba0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7937598Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7937685Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7937829Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7938071Z         """Import a fresh example while proving its module scope is Warp-free."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7938156Z         blocked = {
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7938230Z             "warp",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7938333Z             "particula.gpu",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7938444Z             "particula.execution.gpu_session",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7938558Z             "particula.execution.gpu_resources",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7938674Z             "particula.execution.checkpoint",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7938804Z             "particula.execution.resident_scheduler",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7938887Z         }
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7938990Z         original_import = builtins.__import__
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7939068Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7939154Z         def guarded_import(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7939240Z             name: str,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7939363Z             globals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7939483Z             locals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7939576Z             fromlist: Sequence[str] = (),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7939668Z             level: int = 0,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7939753Z         ) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7939839Z             if name in blocked:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7939972Z                 pytest.fail(f"example imported {name} eagerly")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7940134Z             return original_import(name, globals, locals, fromlist, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7940216Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7940350Z         monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7940493Z         sys.modules.pop("gpu_resident_multi_timestep", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7940653Z         monkeypatch.setattr(builtins, "__import__", guarded_import)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7940809Z >       module = importlib.import_module("gpu_resident_multi_timestep")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7940922Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7940926Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7941074Z particula/tests/gpu_resident_multi_timestep_docs_test.py:67: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7941210Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7941859Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7942004Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7942111Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7942226Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7942306Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7942425Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7942507Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7942641Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7942645Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7942743Z name = 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7942868Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7942872Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7942943Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7943120Z E   ModuleNotFoundError: No module named 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7943125Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7943263Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7943496Z ______ ERROR at setup of test_missing_top_level_warp_runs_no_enabled_work ______
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7943500Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7943693Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0462630>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7943697Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7943778Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7943929Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7944115Z         """Import a fresh example while proving its module scope is Warp-free."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7944192Z         blocked = {
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7944273Z             "warp",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7944364Z             "particula.gpu",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7944523Z             "particula.execution.gpu_session",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7944631Z             "particula.execution.gpu_resources",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7944748Z             "particula.execution.checkpoint",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7944867Z             "particula.execution.resident_scheduler",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7944951Z         }
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7945063Z         original_import = builtins.__import__
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7945136Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7945228Z         def guarded_import(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7945323Z             name: str,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7945448Z             globals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7945559Z             locals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7945664Z             fromlist: Sequence[str] = (),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7945753Z             level: int = 0,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7945831Z         ) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7945923Z             if name in blocked:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7946052Z                 pytest.fail(f"example imported {name} eagerly")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7946301Z             return original_import(name, globals, locals, fromlist, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7946378Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7946521Z         monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7946656Z         sys.modules.pop("gpu_resident_multi_timestep", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7946822Z         monkeypatch.setattr(builtins, "__import__", guarded_import)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7946987Z >       module = importlib.import_module("gpu_resident_multi_timestep")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7947092Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7947097Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7947255Z particula/tests/gpu_resident_multi_timestep_docs_test.py:67: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7947379Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7948033Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7948184Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7948280Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7948403Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7948474Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7948603Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7948675Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7948810Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7948813Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7948919Z name = 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7949032Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7949035Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7949119Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7949422Z E   ModuleNotFoundError: No module named 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7949435Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7949661Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7949945Z _________ ERROR at setup of test_broken_warp_import_propagates[error0] _________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7949952Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7950276Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea04a0920>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7950282Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7950502Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7950714Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7950967Z         """Import a fresh example while proving its module scope is Warp-free."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7951115Z         blocked = {
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7951238Z             "warp",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7951368Z             "particula.gpu",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7951535Z             "particula.execution.gpu_session",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7951719Z             "particula.execution.gpu_resources",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7951907Z             "particula.execution.checkpoint",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7952191Z             "particula.execution.resident_scheduler",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7952305Z         }
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7952482Z         original_import = builtins.__import__
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7952600Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7952729Z         def guarded_import(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7952880Z             name: str,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7953096Z             globals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7953295Z             locals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7953445Z             fromlist: Sequence[str] = (),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7953596Z             level: int = 0,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7953732Z         ) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7953874Z             if name in blocked:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7954098Z                 pytest.fail(f"example imported {name} eagerly")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7954379Z             return original_import(name, globals, locals, fromlist, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7954496Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7954736Z         monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7954975Z         sys.modules.pop("gpu_resident_multi_timestep", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7955245Z         monkeypatch.setattr(builtins, "__import__", guarded_import)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7955550Z >       module = importlib.import_module("gpu_resident_multi_timestep")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7955741Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7955748Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7956032Z particula/tests/gpu_resident_multi_timestep_docs_test.py:67: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7956357Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7957452Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7957715Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7957895Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7958085Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7958190Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7958311Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7958393Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7958523Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7958536Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7958632Z name = 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7958755Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7958759Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7958830Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7959008Z E   ModuleNotFoundError: No module named 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7959012Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7959143Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7959324Z _________ ERROR at setup of test_broken_warp_import_propagates[error1] _________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7959330Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7959514Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0460e90>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7959518Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7959598Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7959750Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7959926Z         """Import a fresh example while proving its module scope is Warp-free."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7960080Z         blocked = {
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7960166Z             "warp",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7960256Z             "particula.gpu",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7960371Z             "particula.execution.gpu_session",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7960479Z             "particula.execution.gpu_resources",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7960596Z             "particula.execution.checkpoint",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7960712Z             "particula.execution.resident_scheduler",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7960793Z         }
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7960894Z         original_import = builtins.__import__
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7961030Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7961121Z         def guarded_import(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7961199Z             name: str,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7961327Z             globals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7961441Z             locals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7961542Z             fromlist: Sequence[str] = (),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7961624Z             level: int = 0,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7961712Z         ) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7961805Z             if name in blocked:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7961932Z                 pytest.fail(f"example imported {name} eagerly")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7962095Z             return original_import(name, globals, locals, fromlist, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7962167Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7962307Z         monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7962441Z         sys.modules.pop("gpu_resident_multi_timestep", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7962601Z         monkeypatch.setattr(builtins, "__import__", guarded_import)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7962768Z >       module = importlib.import_module("gpu_resident_multi_timestep")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7962872Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7962876Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7963031Z particula/tests/gpu_resident_multi_timestep_docs_test.py:67: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7963157Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7963809Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7963958Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7964053Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7964175Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7964245Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7964379Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7964450Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7964583Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7964587Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7964689Z name = 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7964805Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7964812Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7964890Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7965058Z E   ModuleNotFoundError: No module named 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7965062Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7965203Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7965380Z _____ ERROR at setup of test_loader_requests_only_concrete_resident_seams ______
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7965391Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7965571Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea04a3740>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7965574Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7965660Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7965807Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7965991Z         """Import a fresh example while proving its module scope is Warp-free."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7966070Z         blocked = {
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7966255Z             "warp",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7966361Z             "particula.gpu",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7966467Z             "particula.execution.gpu_session",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7966644Z             "particula.execution.gpu_resources",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7966758Z             "particula.execution.checkpoint",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7966887Z             "particula.execution.resident_scheduler",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7966961Z         }
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7967070Z         original_import = builtins.__import__
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7967141Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7967232Z         def guarded_import(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7967321Z             name: str,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7967438Z             globals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7967617Z             locals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7967714Z             fromlist: Sequence[str] = (),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7967803Z             level: int = 0,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7967880Z         ) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7967974Z             if name in blocked:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7968105Z                 pytest.fail(f"example imported {name} eagerly")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7968263Z             return original_import(name, globals, locals, fromlist, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7968341Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7968473Z         monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7968650Z         sys.modules.pop("gpu_resident_multi_timestep", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7968804Z         monkeypatch.setattr(builtins, "__import__", guarded_import)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7968967Z >       module = importlib.import_module("gpu_resident_multi_timestep")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7969078Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7969086Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7969237Z particula/tests/gpu_resident_multi_timestep_docs_test.py:67: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7969366Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7970013Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7970158Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7970261Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7970374Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7970451Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7970571Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7970654Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7970780Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7970787Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7970889Z name = 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7971011Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7971015Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7971089Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7971265Z E   ModuleNotFoundError: No module named 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7971270Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7971407Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7971613Z _ ERROR at setup of test_enabled_loader_error_propagates_without_fixture[error0] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7971616Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7971860Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0473e30>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7971868Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7971991Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7972237Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7972541Z         """Import a fresh example while proving its module scope is Warp-free."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7972665Z         blocked = {
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7972779Z             "warp",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7972935Z             "particula.gpu",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7973115Z             "particula.execution.gpu_session",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7973259Z             "particula.execution.gpu_resources",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7973426Z             "particula.execution.checkpoint",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7973668Z             "particula.execution.resident_scheduler",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7973754Z         }
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7973855Z         original_import = builtins.__import__
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7973936Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7974021Z         def guarded_import(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7974163Z             name: str,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7974375Z             globals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7974558Z             locals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7974713Z             fromlist: Sequence[str] = (),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7974837Z             level: int = 0,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7974961Z         ) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7975158Z             if name in blocked:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7975377Z                 pytest.fail(f"example imported {name} eagerly")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7975577Z             return original_import(name, globals, locals, fromlist, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7975649Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7975785Z         monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7975924Z         sys.modules.pop("gpu_resident_multi_timestep", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7976083Z         monkeypatch.setattr(builtins, "__import__", guarded_import)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7976406Z >       module = importlib.import_module("gpu_resident_multi_timestep")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7976526Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7976531Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7976690Z particula/tests/gpu_resident_multi_timestep_docs_test.py:67: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7976814Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7977467Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7977606Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7977715Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7977838Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7977912Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7978040Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7978115Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7978245Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7978248Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7978343Z name = 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7978466Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7978469Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7978551Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7978719Z E   ModuleNotFoundError: No module named 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7978722Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7978859Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7979049Z _ ERROR at setup of test_enabled_loader_error_propagates_without_fixture[error1] _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7979053Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7979241Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea08ee690>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7979245Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7979333Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7979476Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7979655Z         """Import a fresh example while proving its module scope is Warp-free."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7979732Z         blocked = {
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7979816Z             "warp",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7979903Z             "particula.gpu",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7980018Z             "particula.execution.gpu_session",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7980131Z             "particula.execution.gpu_resources",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7980241Z             "particula.execution.checkpoint",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7980364Z             "particula.execution.resident_scheduler",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7980435Z         }
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7980544Z         original_import = builtins.__import__
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7980617Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7980707Z         def guarded_import(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7980851Z             name: str,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7980981Z             globals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7981097Z             locals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7981189Z             fromlist: Sequence[str] = (),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7981280Z             level: int = 0,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7981355Z         ) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7981446Z             if name in blocked:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7981568Z                 pytest.fail(f"example imported {name} eagerly")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7981732Z             return original_import(name, globals, locals, fromlist, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7981862Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7981995Z         monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7982138Z         sys.modules.pop("gpu_resident_multi_timestep", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7982290Z         monkeypatch.setattr(builtins, "__import__", guarded_import)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7982457Z >       module = importlib.import_module("gpu_resident_multi_timestep")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7982562Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7982574Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7982722Z particula/tests/gpu_resident_multi_timestep_docs_test.py:67: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7982852Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7983490Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7983640Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7983734Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7983858Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7983938Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7984056Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7984137Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7984263Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7984267Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7984368Z name = 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7984481Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7984492Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7984563Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7984735Z E   ModuleNotFoundError: No module named 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7984739Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7984870Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7985058Z ____ ERROR at setup of test_availability_failure_precedes_fixture_and_setup ____
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7985061Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7985238Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea05f3d70>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7985250Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7985328Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7985480Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7985657Z         """Import a fresh example while proving its module scope is Warp-free."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7985743Z         blocked = {
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7985818Z             "warp",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7985912Z             "particula.gpu",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7986018Z             "particula.execution.gpu_session",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7986222Z             "particula.execution.gpu_resources",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7986346Z             "particula.execution.checkpoint",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7986461Z             "particula.execution.resident_scheduler",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7986539Z         }
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7986644Z         original_import = builtins.__import__
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7986720Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7986803Z         def guarded_import(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7986889Z             name: str,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7987014Z             globals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7987125Z             locals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7987285Z             fromlist: Sequence[str] = (),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7987370Z             level: int = 0,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7987453Z         ) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7987537Z             if name in blocked:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7987669Z                 pytest.fail(f"example imported {name} eagerly")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7987822Z             return original_import(name, globals, locals, fromlist, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7987900Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7988038Z         monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7988169Z         sys.modules.pop("gpu_resident_multi_timestep", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7988379Z         monkeypatch.setattr(builtins, "__import__", guarded_import)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7988534Z >       module = importlib.import_module("gpu_resident_multi_timestep")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7988646Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7988650Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7988810Z particula/tests/gpu_resident_multi_timestep_docs_test.py:67: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7988933Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7989579Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7989720Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7989826Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7989943Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7990022Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7990150Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7990224Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7990354Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7990358Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7990454Z name = 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7990576Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7990579Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7990660Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7990828Z E   ModuleNotFoundError: No module named 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7990831Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7990970Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7991161Z _ ERROR at setup of test_setup_failure_propagates_without_checkpoint_or_restart _
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7991165Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7991346Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea078b0b0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7991353Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7991441Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7991586Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7991764Z         """Import a fresh example while proving its module scope is Warp-free."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7991842Z         blocked = {
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7991924Z             "warp",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7992014Z             "particula.gpu",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7992127Z             "particula.execution.gpu_session",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7992234Z             "particula.execution.gpu_resources",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7992350Z             "particula.execution.checkpoint",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7992472Z             "particula.execution.resident_scheduler",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7992548Z         }
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7992656Z         original_import = builtins.__import__
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7992726Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7992816Z         def guarded_import(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7992894Z             name: str,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7993112Z             globals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7993302Z             locals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7993443Z             fromlist: Sequence[str] = (),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7993580Z             level: int = 0,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7993695Z         ) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7993836Z             if name in blocked:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7994166Z                 pytest.fail(f"example imported {name} eagerly")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7994431Z             return original_import(name, globals, locals, fromlist, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7994545Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7994791Z         monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7995032Z         sys.modules.pop("gpu_resident_multi_timestep", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7995293Z         monkeypatch.setattr(builtins, "__import__", guarded_import)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7995557Z >       module = importlib.import_module("gpu_resident_multi_timestep")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7995789Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7995794Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7995958Z particula/tests/gpu_resident_multi_timestep_docs_test.py:67: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7996263Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7997427Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7997579Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7997677Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7997802Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7997875Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7998003Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7998082Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7998212Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7998216Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7998319Z name = 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7998434Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7998438Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7998521Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7998700Z E   ModuleNotFoundError: No module named 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7998704Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7998836Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7999028Z _ ERROR at setup of test_writer_dispatch_failure_propagates_after_guard_close __
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7999032Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7999213Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0470aa0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7999217Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7999305Z     @pytest.fixture
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7999456Z     def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7999630Z         """Import a fresh example while proving its module scope is Warp-free."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7999717Z         blocked = {
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7999792Z             "warp",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7999887Z             "particula.gpu",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.7999995Z             "particula.execution.gpu_session",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8000113Z             "particula.execution.gpu_resources",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8000223Z             "particula.execution.checkpoint",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8000350Z             "particula.execution.resident_scheduler",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8000430Z         }
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8000532Z         original_import = builtins.__import__
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8000611Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8000695Z         def guarded_import(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8000780Z             name: str,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8000898Z             globals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8001020Z             locals: Mapping[str, object] | None = None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8001123Z             fromlist: Sequence[str] = (),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8001205Z             level: int = 0,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8001288Z         ) -> Any:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8001373Z             if name in blocked:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8001505Z                 pytest.fail(f"example imported {name} eagerly")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8001659Z             return original_import(name, globals, locals, fromlist, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8001738Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8001968Z         monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8002115Z         sys.modules.pop("gpu_resident_multi_timestep", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8002274Z         monkeypatch.setattr(builtins, "__import__", guarded_import)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8002430Z >       module = importlib.import_module("gpu_resident_multi_timestep")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8002542Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8002546Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8002696Z particula/tests/gpu_resident_multi_timestep_docs_test.py:67: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8002827Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8003524Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8003667Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8003778Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8003896Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8003982Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8004114Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8004191Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8004324Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8004328Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8004427Z name = 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8004551Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8004557Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8004632Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8004812Z E   ModuleNotFoundError: No module named 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8004816Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8004953Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8005076Z =================================== FAILURES ===================================
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8005264Z ________ test_condensation_latent_heat_example_runs_as_main_entrypoint _________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8005267Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8005436Z capsys = <_pytest.capture.CaptureFixture object at 0x7f3eadc98a10>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8005440Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8005612Z     def test_condensation_latent_heat_example_runs_as_main_entrypoint(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8005699Z         capsys,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8005783Z     ) -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8005961Z         """Published example path runs as ``__main__`` and prints labels."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8006101Z >       runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8006205Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8006440Z particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py:65: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8006559Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8006658Z <frozen runpy>:286: in run_path
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8006742Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8006863Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8006866Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8007075Z fname = '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8007079Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8007151Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8007495Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8007499Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8007607Z <frozen runpy>:254: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8007786Z ___________ test_condensation_latent_heat_example_runs_without_pint ____________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8007792Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8007980Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3eadc99c40>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8007984Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8008130Z     def test_condensation_latent_heat_example_runs_without_pint(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8008221Z         monkeypatch,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8008374Z     ) -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8008559Z         """Already-SI example inputs do not require the optional Pint package."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8008757Z         monkeypatch.setattr("particula.util.convert_units.unit_registry", None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8008829Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8008955Z >       namespace = runpy.run_path(str(EXAMPLE_PATH))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8009048Z                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8009052Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8009280Z particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py:116: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8009408Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8009862Z <frozen runpy>:286: in run_path
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8009941Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8010058Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8010062Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8010266Z fname = '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8010270Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8010349Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8010681Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8010686Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8010793Z <frozen runpy>:254: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8010978Z ___ test_condensation_latent_heat_run_example_adds_zero_transfer_explanation ___
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8010982Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8011197Z     def test_condensation_latent_heat_run_example_adds_zero_transfer_explanation() -> (
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8011276Z         None
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8011355Z     ):
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8011524Z         """Zero-transfer branch adds the documented explanation string."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8011698Z >       namespace = cast(ExampleNamespace, runpy.run_path(str(EXAMPLE_PATH)))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8011814Z                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8011818Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8012033Z particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py:219: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8012163Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8012255Z <frozen runpy>:286: in run_path
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8012327Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8012452Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8012456Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8012646Z fname = '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8012649Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8012731Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8013055Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8013067Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8013164Z <frozen runpy>:254: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8013356Z _____ test_condensation_latent_heat_main_prints_zero_transfer_explanation ______
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8013362Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8013529Z capsys = <_pytest.capture.CaptureFixture object at 0x7f3eae094ad0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8013532Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8013717Z     def test_condensation_latent_heat_main_prints_zero_transfer_explanation(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8013800Z         capsys,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8013878Z     ) -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8014053Z         """Main prints the zero-transfer explanation when present in results."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8014224Z >       namespace = cast(ExampleNamespace, runpy.run_path(str(EXAMPLE_PATH)))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8014410Z                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8014424Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8014796Z particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py:236: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8015008Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8015155Z <frozen runpy>:286: in run_path
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8015267Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8015568Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8015577Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8015946Z fname = '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8015952Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8016079Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8016791Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8016800Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8016966Z <frozen runpy>:254: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8017272Z _________________ test_forced_disabled_script_has_exact_stdout _________________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8017348Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8017593Z     def test_forced_disabled_script_has_exact_stdout() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8017884Z         """The standalone forced-disabled command exits successfully."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8018040Z >       result = subprocess.run(  # noqa: S603
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8018156Z             [sys.executable, str(_EXAMPLE)],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8018279Z             check=True,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8018429Z             capture_output=True,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8018564Z             text=True,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8018811Z             env={**os.environ, "PARTICULA_EXAMPLE_FORCE_NO_WARP": "1"},
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8018949Z             timeout=10,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8019064Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8019071Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8019348Z particula/execution/tests/gpu_resident_session_example_test.py:123: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8019480Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8019489Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8019634Z input = None, capture_output = True, timeout = 10, check = True
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8020121Z popenargs = (['/home/conda/feedstock_root/build_artifacts/particula_1791096598735/_test_env_placehold_placehold_placehold_placehol... '$SRC_DIR/docs/Examples/gpu_resident_session.py'],)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8020580Z kwargs = {'text': True, 'env': {'DISTRO_ARCH': 'amd64', 'UPLOAD_PACKAGES': 'true', 'CONDA_LIBMAMBA_SOLVER_NO_CHANNELS_FROM_INSTALLED': '1', 'BINSTAR_TOKEN': '', ...}, 'stdout': -1, 'stderr': -1}
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8020592Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8020680Z     def run(*popenargs,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8020873Z             input=None, capture_output=False, timeout=None, check=False, **kwargs):
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8021039Z         """Run command with arguments and return a CompletedProcess instance.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8021117Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8021294Z         The returned instance will have attributes args, returncode, stdout and
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8021490Z         stderr. By default, stdout and stderr are not captured, and those attributes
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8021681Z         will be None. Pass stdout=PIPE and/or stderr=PIPE in order to capture them,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8021797Z         or pass capture_output=True to capture both.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8021875Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8022024Z         If check is True and the exit code was non-zero, it raises a
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8022221Z         CalledProcessError. The CalledProcessError object will have the return code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8022399Z         in the returncode attribute, and output & stderr attributes if those streams
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8022489Z         were captured.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8022570Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8022716Z         If timeout (seconds) is given and the process takes too long,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8022836Z          a TimeoutExpired exception will be raised.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8022908Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8023050Z         There is an optional argument "input", allowing you to
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8023227Z         pass bytes or a string to the subprocess's stdin.  If you use this argument
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8023391Z         you may not also use the Popen constructor's "stdin" argument, as
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8023493Z         it will be used internally.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8023564Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8023803Z         By default, all communication is in bytes, and therefore any "input" should
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8023973Z         be bytes, and the stdout and stderr will be bytes. If in text mode, any
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8024150Z         "input" should be a string, and stdout and stderr will be strings decoded
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8024315Z         according to locale encoding, or by "encoding" if set. Text mode is
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8024494Z         triggered by setting any of text, encoding, errors or universal_newlines.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8024573Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8024720Z         The other arguments are the same as for the Popen constructor.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8024844Z         """
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8024933Z         if input is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8025042Z             if kwargs.get('stdin') is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8025208Z                 raise ValueError('stdin and input arguments may not both be used.')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8025306Z             kwargs['stdin'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8025383Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8025470Z         if capture_output:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8025646Z             if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8025804Z                 raise ValueError('stdout and stderr arguments may not be used '
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8025917Z                                  'with capture_output.')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8026005Z             kwargs['stdout'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8026101Z             kwargs['stderr'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8026277Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8026409Z         with Popen(*popenargs, **kwargs) as process:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8026493Z             try:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8026651Z                 stdout, stderr = process.communicate(input, timeout=timeout)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8026755Z             except TimeoutExpired as exc:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8026845Z                 process.kill()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8026936Z                 if _mswindows:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8027079Z                     # Windows accumulates the output in a single blocking
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8027220Z                     # read() call run on child threads, with the timeout
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8027373Z                     # being done in a join() on those threads.  communicate()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8027513Z                     # _after_ kill() is required to collect that and add it
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8027614Z                     # to the exception.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8027750Z                     exc.stdout, exc.stderr = process.communicate()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8027840Z                 else:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8027977Z                     # POSIX _communicate already populated the output so
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8028107Z                     # far into the TimeoutExpired exception.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8028205Z                     process.wait()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8028283Z                 raise
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8028453Z             except:  # Including KeyboardInterrupt, communicate handled that.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8028540Z                 process.kill()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8028694Z                 # We don't call process.wait() as .__exit__ does that for us.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8028770Z                 raise
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8028873Z             retcode = process.poll()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8028971Z             if check and retcode:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8029103Z >               raise CalledProcessError(retcode, process.args,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8029230Z                                          output=stdout, stderr=stderr)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8029650Z E               subprocess.CalledProcessError: Command '['$PREFIX/bin/python3.12', '$SRC_DIR/docs/Examples/gpu_resident_session.py']' returned non-zero exit status 2.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8029657Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8030299Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/subprocess.py:571: CalledProcessError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8030524Z _____________________ test_real_warp_cpu_lifecycle_example _____________________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8030528Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8030656Z     def test_real_warp_cpu_lifecycle_example() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8030811Z         """The real Warp CPU route preserves the identity lifecycle."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8030909Z         pytest.importorskip("warp")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8031041Z         spec = importlib.util.spec_from_file_location(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8031148Z             "resident_example_real", _EXAMPLE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8031220Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8031353Z         assert spec is not None and spec.loader is not None
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8031523Z         module = importlib.util.module_from_spec(spec)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8031624Z         sys.modules[spec.name] = module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8031697Z         try:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8031802Z >           spec.loader.exec_module(module)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8031807Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8031988Z particula/execution/tests/gpu_resident_session_example_test.py:217: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8032115Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8032261Z <frozen importlib._bootstrap_external>:995: in exec_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8032337Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8032476Z <frozen importlib._bootstrap_external>:1132: in get_code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8032550Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8032683Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8032686Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8032879Z self = <_frozen_importlib_external.SourceFileLoader object at 0x7f3ea4b01940>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8033013Z path = '$SRC_DIR/docs/Examples/gpu_resident_session.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8033017Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8033098Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8033367Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/gpu_resident_session.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8033372Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8033531Z <frozen importlib._bootstrap_external>:1190: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8033704Z _____________________ test_example_runs_as_main_entrypoint _____________________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8033708Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8033890Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea08b6d20>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8034063Z capsys = <_pytest.capture.CaptureFixture object at 0x7f3ea08b6ba0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8034067Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8034177Z     def test_example_runs_as_main_entrypoint(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8034287Z         monkeypatch: pytest.MonkeyPatch,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8034391Z         capsys: pytest.CaptureFixture[str],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8034480Z     ) -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8034648Z         """Test the published example executes successfully as __main__."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8034801Z         monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8034880Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8035015Z >       runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8035018Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8035170Z particula/gpu/tests/data_containers_example_test.py:204: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8035292Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8035390Z <frozen runpy>:286: in run_path
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8035470Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8035591Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8035595Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8035766Z fname = '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8035769Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8035880Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8036516Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8036531Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8036712Z <frozen runpy>:254: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8036986Z ______________________ test_guide_runs_as_main_entrypoint ______________________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8036992Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8037377Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea08b7860>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8037680Z capsys = <_pytest.capture.CaptureFixture object at 0x7f3ea08b76e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8037690Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8038214Z     def test_guide_runs_as_main_entrypoint(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8038321Z         monkeypatch: pytest.MonkeyPatch,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8038421Z         capsys: pytest.CaptureFixture[str],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8038509Z     ) -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8038670Z         """Test the guide-local forwarding module executes successfully."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8038854Z         monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8039049Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8039279Z >       runpy.run_path(str(GUIDE_PATH), run_name="__main__")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8039285Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8039529Z particula/gpu/tests/data_containers_example_test.py:217: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8039738Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8039894Z <frozen runpy>:286: in run_path
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8040004Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8040224Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8040230Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8040561Z fname = '$SRC_DIR/docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8040566Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8040639Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8040998Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8041002Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8041103Z <frozen runpy>:254: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8041283Z ________________ test_example_non_warp_path_reports_cpu_success ________________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8041286Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8041444Z     def test_example_non_warp_path_reports_cpu_success() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8041618Z         """Test the published example path completes without Warp transfers."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8041738Z >       result = _run_example(force_no_warp=True)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8041830Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8041834Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8041980Z particula/gpu/tests/data_containers_example_test.py:225: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8042104Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8042288Z particula/gpu/tests/data_containers_example_test.py:87: in _run_example
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8042489Z     return subprocess.run(  # noqa: S603 - repo-local example path and interpreter
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8042609Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8042616Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8042775Z input = None, capture_output = True, timeout = None, check = True
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8043444Z popenargs = (['/home/conda/feedstock_root/build_artifacts/particula_1791096598735/_test_env_placehold_placehold_placehold_placehol...eedstock_root/build_artifacts/particula_1791096598735/test_tmp/docs/Examples/data_containers_and_gpu_foundations.py'],)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8043906Z kwargs = {'text': True, 'env': {'DISTRO_ARCH': 'amd64', 'UPLOAD_PACKAGES': 'true', 'CONDA_LIBMAMBA_SOLVER_NO_CHANNELS_FROM_INSTALLED': '1', 'BINSTAR_TOKEN': '', ...}, 'stdout': -1, 'stderr': -1}
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8043910Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8044004Z     def run(*popenargs,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8044183Z             input=None, capture_output=False, timeout=None, check=False, **kwargs):
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8044356Z         """Run command with arguments and return a CompletedProcess instance.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8044427Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8044614Z         The returned instance will have attributes args, returncode, stdout and
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8044805Z         stderr. By default, stdout and stderr are not captured, and those attributes
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8044984Z         will be None. Pass stdout=PIPE and/or stderr=PIPE in order to capture them,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8045107Z         or pass capture_output=True to capture both.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8045229Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8045384Z         If check is True and the exit code was non-zero, it raises a
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8045573Z         CalledProcessError. The CalledProcessError object will have the return code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8045761Z         in the returncode attribute, and output & stderr attributes if those streams
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8045863Z         were captured.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8045933Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8046087Z         If timeout (seconds) is given and the process takes too long,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8046304Z          a TimeoutExpired exception will be raised.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8046503Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8046642Z         There is an optional argument "input", allowing you to
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8046826Z         pass bytes or a string to the subprocess's stdin.  If you use this argument
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8046995Z         you may not also use the Popen constructor's "stdin" argument, as
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8047089Z         it will be used internally.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8047171Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8047350Z         By default, all communication is in bytes, and therefore any "input" should
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8047525Z         be bytes, and the stdout and stderr will be bytes. If in text mode, any
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8047698Z         "input" should be a string, and stdout and stderr will be strings decoded
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8047871Z         according to locale encoding, or by "encoding" if set. Text mode is
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8048054Z         triggered by setting any of text, encoding, errors or universal_newlines.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8048130Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8048289Z         The other arguments are the same as for the Popen constructor.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8048365Z         """
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8048464Z         if input is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8048568Z             if kwargs.get('stdin') is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8048743Z                 raise ValueError('stdin and input arguments may not both be used.')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8048848Z             kwargs['stdin'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8048917Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8049010Z         if capture_output:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8049179Z             if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8049342Z                 raise ValueError('stdout and stderr arguments may not be used '
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8049451Z                                  'with capture_output.')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8049547Z             kwargs['stdout'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8049642Z             kwargs['stderr'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8049715Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8049838Z         with Popen(*popenargs, **kwargs) as process:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8049912Z             try:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8050074Z                 stdout, stderr = process.communicate(input, timeout=timeout)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8050171Z             except TimeoutExpired as exc:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8050269Z                 process.kill()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8050363Z                 if _mswindows:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8050502Z                     # Windows accumulates the output in a single blocking
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8050635Z                     # read() call run on child threads, with the timeout
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8050776Z                     # being done in a join() on those threads.  communicate()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8050918Z                     # _after_ kill() is required to collect that and add it
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8051005Z                     # to the exception.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8051144Z                     exc.stdout, exc.stderr = process.communicate()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8051232Z                 else:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8051365Z                     # POSIX _communicate already populated the output so
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8051486Z                     # far into the TimeoutExpired exception.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8051575Z                     process.wait()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8051658Z                 raise
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8051877Z             except:  # Including KeyboardInterrupt, communicate handled that.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8051973Z                 process.kill()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8052127Z                 # We don't call process.wait() as .__exit__ does that for us.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8052203Z                 raise
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8052303Z             retcode = process.poll()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8052390Z             if check and retcode:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8052528Z >               raise CalledProcessError(retcode, process.args,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8052643Z                                          output=stdout, stderr=stderr)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8053101Z E               subprocess.CalledProcessError: Command '['$PREFIX/bin/python3.12', '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py']' returned non-zero exit status 2.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8053150Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8053793Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/subprocess.py:571: CalledProcessError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8053977Z __________ test_example_warp_path_reports_round_trip_shapes_and_names __________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8053980Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8054161Z     @pytest.mark.skipif(not WARP_AVAILABLE, reason="Warp is not available")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8054334Z     def test_example_warp_path_reports_round_trip_shapes_and_names() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8054505Z         """Test the published example path exercises Warp CPU round trips."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8054622Z >       result = _run_example(force_no_warp=False)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8054716Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8054720Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8054868Z particula/gpu/tests/data_containers_example_test.py:264: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8054994Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8055174Z particula/gpu/tests/data_containers_example_test.py:87: in _run_example
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8055376Z     return subprocess.run(  # noqa: S603 - repo-local example path and interpreter
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8055498Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8055501Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8055659Z input = None, capture_output = True, timeout = None, check = True
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8056422Z popenargs = (['/home/conda/feedstock_root/build_artifacts/particula_1791096598735/_test_env_placehold_placehold_placehold_placehol...eedstock_root/build_artifacts/particula_1791096598735/test_tmp/docs/Examples/data_containers_and_gpu_foundations.py'],)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8056899Z kwargs = {'text': True, 'env': {'DISTRO_ARCH': 'amd64', 'UPLOAD_PACKAGES': 'true', 'CONDA_LIBMAMBA_SOLVER_NO_CHANNELS_FROM_INSTALLED': '1', 'BINSTAR_TOKEN': '', ...}, 'stdout': -1, 'stderr': -1}
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8056904Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8056996Z     def run(*popenargs,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8057177Z             input=None, capture_output=False, timeout=None, check=False, **kwargs):
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8057349Z         """Run command with arguments and return a CompletedProcess instance.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8057420Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8057604Z         The returned instance will have attributes args, returncode, stdout and
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8057893Z         stderr. By default, stdout and stderr are not captured, and those attributes
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8058209Z         will be None. Pass stdout=PIPE and/or stderr=PIPE in order to capture them,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8058398Z         or pass capture_output=True to capture both.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8058506Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8058790Z         If check is True and the exit code was non-zero, it raises a
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8059083Z         CalledProcessError. The CalledProcessError object will have the return code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8059389Z         in the returncode attribute, and output & stderr attributes if those streams
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8059487Z         were captured.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8059561Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8059785Z         If timeout (seconds) is given and the process takes too long,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8059901Z          a TimeoutExpired exception will be raised.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8059983Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8060119Z         There is an optional argument "input", allowing you to
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8060303Z         pass bytes or a string to the subprocess's stdin.  If you use this argument
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8060470Z         you may not also use the Popen constructor's "stdin" argument, as
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8060565Z         it will be used internally.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8060699Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8060877Z         By default, all communication is in bytes, and therefore any "input" should
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8061048Z         be bytes, and the stdout and stderr will be bytes. If in text mode, any
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8061215Z         "input" should be a string, and stdout and stderr will be strings decoded
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8061389Z         according to locale encoding, or by "encoding" if set. Text mode is
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8061571Z         triggered by setting any of text, encoding, errors or universal_newlines.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8061642Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8061822Z         The other arguments are the same as for the Popen constructor.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8061934Z         """
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8062083Z         if input is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8062245Z             if kwargs.get('stdin') is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8062544Z                 raise ValueError('stdin and input arguments may not both be used.')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8062692Z             kwargs['stdin'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8062802Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8062945Z         if capture_output:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8063236Z             if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8063516Z                 raise ValueError('stdout and stderr arguments may not be used '
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8063693Z                                  'with capture_output.')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8063845Z             kwargs['stdout'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8063981Z             kwargs['stderr'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8064089Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8064272Z         with Popen(*popenargs, **kwargs) as process:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8064392Z             try:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8064566Z                 stdout, stderr = process.communicate(input, timeout=timeout)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8064669Z             except TimeoutExpired as exc:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8064768Z                 process.kill()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8064851Z                 if _mswindows:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8065003Z                     # Windows accumulates the output in a single blocking
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8065141Z                     # read() call run on child threads, with the timeout
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8065287Z                     # being done in a join() on those threads.  communicate()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8065434Z                     # _after_ kill() is required to collect that and add it
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8065528Z                     # to the exception.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8065672Z                     exc.stdout, exc.stderr = process.communicate()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8065750Z                 else:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8065897Z                     # POSIX _communicate already populated the output so
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8066019Z                     # far into the TimeoutExpired exception.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8066201Z                     process.wait()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8066298Z                 raise
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8066460Z             except:  # Including KeyboardInterrupt, communicate handled that.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8066558Z                 process.kill()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8066703Z                 # We don't call process.wait() as .__exit__ does that for us.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8066785Z                 raise
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8066886Z             retcode = process.poll()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8066975Z             if check and retcode:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8067177Z >               raise CalledProcessError(retcode, process.args,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8067300Z                                          output=stdout, stderr=stderr)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8067766Z E               subprocess.CalledProcessError: Command '['$PREFIX/bin/python3.12', '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py']' returned non-zero exit status 2.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8067772Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8068450Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/subprocess.py:571: CalledProcessError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8068697Z _____________ test_forced_disabled_routes_never_import_gpu_runtime _____________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8068701Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8068890Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea08ee060>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8069052Z capsys = <_pytest.capture.CaptureFixture object at 0x7f3ea08ed7c0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8069056Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8069206Z     def test_forced_disabled_routes_never_import_gpu_runtime(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8069318Z         monkeypatch: pytest.MonkeyPatch,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8069423Z         capsys: pytest.CaptureFixture[str],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8069510Z     ) -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8069690Z         """Test forced no-Warp behavior avoids conversion and adapter dispatch."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8069826Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8069976Z         monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8070174Z         monkeypatch.delitem(sys.modules, "gpu_coagulation_direct", raising=False)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8070292Z         with monkeypatch.context() as cleanup:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8070389Z             for module_name in LAZY_IMPORTS:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8070542Z                 cleanup.delitem(sys.modules, module_name, raising=False)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8070688Z >           module = importlib.import_module("gpu_coagulation_direct")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8070802Z                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8070806Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8070970Z particula/gpu/tests/gpu_coagulation_direct_example_test.py:79: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8071098Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8071752Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8071897Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8072007Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8072130Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8072205Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8072335Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8072410Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8072546Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8072550Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8072641Z name = 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8072764Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8072768Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8072846Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8073004Z E   ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8073008Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8073146Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8073327Z _______ test_forced_disabled_script_is_a_successful_no_kernel_subprocess _______
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8073333Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8073529Z     def test_forced_disabled_script_is_a_successful_no_kernel_subprocess() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8073724Z         """The standalone disabled route succeeds without an optional Warp import."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8073810Z         example_path = (
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8073991Z             Path(__file__).resolve().parents[3]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8074073Z             / "docs"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8074162Z             / "Examples"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8074268Z             / "gpu_complete_process_sequence.py"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8074351Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8074424Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8074539Z >       process = subprocess.run(  # noqa: S603
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8074647Z             [sys.executable, str(example_path)],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8074728Z             check=True,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8074824Z             capture_output=True,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8074902Z             text=True,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8075102Z             env={**os.environ, "PARTICULA_EXAMPLE_FORCE_NO_WARP": "1"},
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8075183Z             timeout=10,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8075264Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8075268Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8075452Z particula/gpu/tests/gpu_complete_process_sequence_example_test.py:97: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8075574Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8075578Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8075730Z input = None, capture_output = True, timeout = 10, check = True
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8076475Z popenargs = (['/home/conda/feedstock_root/build_artifacts/particula_1791096598735/_test_env_placehold_placehold_placehold_placehol...onda/feedstock_root/build_artifacts/particula_1791096598735/test_tmp/docs/Examples/gpu_complete_process_sequence.py'],)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8076942Z kwargs = {'text': True, 'env': {'DISTRO_ARCH': 'amd64', 'UPLOAD_PACKAGES': 'true', 'CONDA_LIBMAMBA_SOLVER_NO_CHANNELS_FROM_INSTALLED': '1', 'BINSTAR_TOKEN': '', ...}, 'stdout': -1, 'stderr': -1}
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8076951Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8077043Z     def run(*popenargs,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8077221Z             input=None, capture_output=False, timeout=None, check=False, **kwargs):
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8077392Z         """Run command with arguments and return a CompletedProcess instance.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8077462Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8077649Z         The returned instance will have attributes args, returncode, stdout and
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8077833Z         stderr. By default, stdout and stderr are not captured, and those attributes
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8078010Z         will be None. Pass stdout=PIPE and/or stderr=PIPE in order to capture them,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8078132Z         or pass capture_output=True to capture both.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8078203Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8078350Z         If check is True and the exit code was non-zero, it raises a
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8078541Z         CalledProcessError. The CalledProcessError object will have the return code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8078732Z         in the returncode attribute, and output & stderr attributes if those streams
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8078820Z         were captured.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8078889Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8079041Z         If timeout (seconds) is given and the process takes too long,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8079152Z          a TimeoutExpired exception will be raised.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8079233Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8079366Z         There is an optional argument "input", allowing you to
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8079545Z         pass bytes or a string to the subprocess's stdin.  If you use this argument
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8079708Z         you may not also use the Popen constructor's "stdin" argument, as
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8079800Z         it will be used internally.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8079877Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8080052Z         By default, all communication is in bytes, and therefore any "input" should
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8080220Z         be bytes, and the stdout and stderr will be bytes. If in text mode, any
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8080391Z         "input" should be a string, and stdout and stderr will be strings decoded
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8080561Z         according to locale encoding, or by "encoding" if set. Text mode is
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8080744Z         triggered by setting any of text, encoding, errors or universal_newlines.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8080814Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8081156Z         The other arguments are the same as for the Popen constructor.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8081272Z         """
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8081423Z         if input is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8081569Z             if kwargs.get('stdin') is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8081869Z                 raise ValueError('stdin and input arguments may not both be used.')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8082020Z             kwargs['stdin'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8082128Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8082266Z         if capture_output:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8082530Z             if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8082771Z                 raise ValueError('stdout and stderr arguments may not be used '
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8082882Z                                  'with capture_output.')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8082981Z             kwargs['stdout'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8083073Z             kwargs['stderr'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8083217Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8083421Z         with Popen(*popenargs, **kwargs) as process:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8083525Z             try:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8083821Z                 stdout, stderr = process.communicate(input, timeout=timeout)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8083980Z             except TimeoutExpired as exc:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8084097Z                 process.kill()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8084184Z                 if _mswindows:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8084423Z                     # Windows accumulates the output in a single blocking
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8084572Z                     # read() call run on child threads, with the timeout
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8084721Z                     # being done in a join() on those threads.  communicate()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8084867Z                     # _after_ kill() is required to collect that and add it
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8084991Z                     # to the exception.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8085231Z                     exc.stdout, exc.stderr = process.communicate()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8085355Z                 else:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8085568Z                     # POSIX _communicate already populated the output so
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8085695Z                     # far into the TimeoutExpired exception.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8085786Z                     process.wait()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8085871Z                 raise
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8086035Z             except:  # Including KeyboardInterrupt, communicate handled that.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8086264Z                 process.kill()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8086429Z                 # We don't call process.wait() as .__exit__ does that for us.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8086518Z                 raise
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8086618Z             retcode = process.poll()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8086707Z             if check and retcode:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8086846Z >               raise CalledProcessError(retcode, process.args,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8086962Z                                          output=stdout, stderr=stderr)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8087416Z E               subprocess.CalledProcessError: Command '['$PREFIX/bin/python3.12', '$SRC_DIR/docs/Examples/gpu_complete_process_sequence.py']' returned non-zero exit status 2.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8087422Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8088097Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/subprocess.py:571: CalledProcessError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8088408Z _______________ test_main_force_no_warp_prints_oracle_completion _______________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8088415Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8088746Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07ea630>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8089023Z capsys = <_pytest.capture.CaptureFixture object at 0x7f3ea07b5640>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8089029Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8089239Z     def test_main_force_no_warp_prints_oracle_completion(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8089565Z         monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8089765Z     ) -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8090076Z         """Direct script execution preserves the CPU-only walkthrough route."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8090332Z         monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8090514Z         with pytest.raises(SystemExit) as error:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8090736Z >           runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8090742Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8091058Z particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py:776: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8091194Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8091342Z <frozen runpy>:286: in run_path
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8091424Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8091546Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8091550Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8091728Z fname = '$SRC_DIR/docs/Examples/gpu_condensation_parity_walkthrough.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8091731Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8091813Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8092127Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/gpu_condensation_parity_walkthrough.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8092131Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8092238Z <frozen runpy>:254: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8092418Z _______ test_forced_no_warp_import_run_main_and_subprocess_defer_kernels _______
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8092421Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8092606Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea078a8a0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8092773Z capsys = <_pytest.capture.CaptureFixture object at 0x7f3ea0788080>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8092780Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8092943Z     def test_forced_no_warp_import_run_main_and_subprocess_defer_kernels(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8093051Z         monkeypatch: pytest.MonkeyPatch,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8093157Z         capsys: pytest.CaptureFixture[str],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8093241Z     ) -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8093416Z         """Test every forced no-Warp route avoids direct and concrete modules."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8093553Z         monkeypatch.syspath_prepend(str(EXAMPLES_ROOT))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8093705Z         monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8093797Z         preexisting_modules = {
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8093934Z             module_name: importlib.import_module(module_name)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8094032Z             for module_name in KERNEL_MODULES
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8094111Z         }
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8094204Z         monkeypatch.delitem(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8094363Z             sys.modules, "gpu_direct_kernels_quick_start", raising=False
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8094447Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8094517Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8094659Z         with monkeypatch.context() as kernel_module_cleanup:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8094760Z             for module_name in KERNEL_MODULES:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8094912Z                 kernel_module_cleanup.delitem(sys.modules, module_name)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8094984Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8095162Z >           module = importlib.import_module("gpu_direct_kernels_quick_start")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8095271Z                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8095285Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8095436Z particula/gpu/tests/gpu_direct_kernels_example_test.py:165: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8095568Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8096323Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8096492Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8096602Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8096719Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8096799Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8096985Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8097071Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8097198Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8097202Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8097312Z name = 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8097437Z import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8097441Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8097514Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8097702Z E   ModuleNotFoundError: No module named 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8097706Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8097897Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8098077Z __________________ test_main_entrypoint_prints_result_output ___________________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8098081Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8098258Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07c3d40>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8098423Z capsys = <_pytest.capture.CaptureFixture object at 0x7f3ea07c3200>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8098429Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8098551Z     def test_main_entrypoint_prints_result_output(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8098652Z         monkeypatch: pytest.MonkeyPatch,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8098762Z         capsys: pytest.CaptureFixture[str],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8098839Z     ) -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8099008Z         """Test runpy execution prints the ``ExampleRun.output`` lines."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8099162Z         monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8099296Z >       runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8099303Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8099457Z particula/gpu/tests/gpu_direct_kernels_example_test.py:256: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8099580Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8099679Z <frozen runpy>:286: in run_path
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8099751Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8099881Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8099887Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8100046Z fname = '$SRC_DIR/docs/Examples/gpu_direct_kernels_quick_start.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8100049Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8100121Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8100411Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/gpu_direct_kernels_quick_start.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8100415Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8100513Z <frozen runpy>:254: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8100701Z _______ test_direct_example_runs_with_explicit_transfer_and_conservation _______
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8100705Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8100800Z     @pytest.mark.warp
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8100990Z     def test_direct_example_runs_with_explicit_transfer_and_conservation() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8101165Z         """Run the standalone fixture and check identity-derived final state."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8101247Z         _require_warp()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8101373Z >       namespace = runpy.run_path(str(EXAMPLE_PATH))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8101469Z                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8101479Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8101632Z particula/gpu/tests/gpu_direct_nucleation_example_test.py:49: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8101758Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8101845Z <frozen runpy>:286: in run_path
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8101923Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8102041Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8102045Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8102213Z fname = '$SRC_DIR/docs/Examples/Nucleation/gpu_direct_nucleation.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8102220Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8102297Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8102584Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Nucleation/gpu_direct_nucleation.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8102588Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8102692Z <frozen runpy>:254: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8102912Z __________ test_direct_example_source_uses_only_documented_boundaries __________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8102916Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8103105Z     def test_direct_example_source_uses_only_documented_boundaries() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8103285Z         """Keep explicit synchronization and concrete-record imports visible."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8103412Z >       source = EXAMPLE_PATH.read_text(encoding="utf-8")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8103517Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8103520Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8103675Z particula/gpu/tests/gpu_direct_nucleation_example_test.py:63: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8103802Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8104513Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/pathlib.py:1027: in read_text
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8104666Z     with self.open(mode='r', encoding=encoding, errors=errors) as f:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8104774Z          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8104893Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8104897Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8105095Z self = PosixPath('$SRC_DIR/docs/Examples/Nucleation/gpu_direct_nucleation.py')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8105299Z mode = 'r', buffering = -1, encoding = 'utf-8', errors = None, newline = None
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8105302Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8105430Z     def open(self, mode='r', buffering=-1, encoding=None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8105534Z              errors=None, newline=None):
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8105641Z         """
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8105918Z         Open the file pointed to by this path and return a file object, as
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8106075Z         the built-in open() function does.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8106289Z         """
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8106440Z         if "b" not in mode:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8106586Z             encoding = io.text_encoding(encoding)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8106854Z >       return io.open(self, mode, buffering, encoding, errors, newline)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8107033Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8107538Z E       FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Nucleation/gpu_direct_nucleation.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8107545Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8108569Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/pathlib.py:1013: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8108899Z _________________ test_documented_direct_example_command_runs __________________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8108910Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8109069Z     @pytest.mark.warp
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8109306Z     def test_documented_direct_example_command_runs() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8109670Z         """Run the documented warning-clean command without cross-process identity."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8109799Z         _require_warp()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8109919Z         completed = _run_documented_command()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8110129Z >       assert completed.returncode == 0, completed.stderr
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8110752Z E       AssertionError: $PREFIX/bin/python3.12: can't open file '$SRC_DIR/docs/Examples/Nucleation/gpu_direct_nucleation.py': [Errno 2] No such file or directory
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8110881Z E         
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8111003Z E       assert 2 == 0
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8112333Z E        +  where 2 = CompletedProcess(args=['/home/conda/feedstock_root/build_artifacts/particula_1791096598735/_test_env_placehold_placeho...cula_1791096598735/test_tmp/docs/Examples/Nucleation/gpu_direct_nucleation.py': [Errno 2] No such file or directory\n").returncode
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8112352Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8112708Z particula/gpu/tests/gpu_direct_nucleation_example_test.py:143: AssertionError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8113020Z _______________________ test_example_forced_no_warp_path _______________________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8113030Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8113244Z     def test_example_forced_no_warp_path() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8113560Z         """The example exits successfully when Warp is explicitly disabled."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8113745Z >       process = subprocess.run(  # noqa: S603
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8113925Z             [sys.executable, str(EXAMPLE_PATH)],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8114006Z             check=True,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8114104Z             capture_output=True,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8114183Z             text=True,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8114466Z             env={**os.environ, "PARTICULA_EXAMPLE_FORCE_NO_WARP": "1"},
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8114601Z             timeout=10,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8114724Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8114731Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8115025Z particula/tests/backend_selected_coagulation_example_test.py:20: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8115240Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8115252Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8115507Z input = None, capture_output = True, timeout = 10, check = True
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8116000Z popenargs = (['/home/conda/feedstock_root/build_artifacts/particula_1791096598735/_test_env_placehold_placehold_placehold_placehol...$SRC_DIR/docs/Examples/gpu_coagulation_direct.py'],)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8116636Z kwargs = {'text': True, 'env': {'DISTRO_ARCH': 'amd64', 'UPLOAD_PACKAGES': 'true', 'CONDA_LIBMAMBA_SOLVER_NO_CHANNELS_FROM_INSTALLED': '1', 'BINSTAR_TOKEN': '', ...}, 'stdout': -1, 'stderr': -1}
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8116645Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8116746Z     def run(*popenargs,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8116926Z             input=None, capture_output=False, timeout=None, check=False, **kwargs):
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8117102Z         """Run command with arguments and return a CompletedProcess instance.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8117174Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8117359Z         The returned instance will have attributes args, returncode, stdout and
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8117542Z         stderr. By default, stdout and stderr are not captured, and those attributes
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8117733Z         will be None. Pass stdout=PIPE and/or stderr=PIPE in order to capture them,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8117859Z         or pass capture_output=True to capture both.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8117932Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8118085Z         If check is True and the exit code was non-zero, it raises a
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8118277Z         CalledProcessError. The CalledProcessError object will have the return code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8118465Z         in the returncode attribute, and output & stderr attributes if those streams
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8118549Z         were captured.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8118628Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8118785Z         If timeout (seconds) is given and the process takes too long,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8118898Z          a TimeoutExpired exception will be raised.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8118974Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8119110Z         There is an optional argument "input", allowing you to
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8119290Z         pass bytes or a string to the subprocess's stdin.  If you use this argument
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8119446Z         you may not also use the Popen constructor's "stdin" argument, as
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8119544Z         it will be used internally.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8119622Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8119798Z         By default, all communication is in bytes, and therefore any "input" should
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8119969Z         be bytes, and the stdout and stderr will be bytes. If in text mode, any
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8120138Z         "input" should be a string, and stdout and stderr will be strings decoded
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8120313Z         according to locale encoding, or by "encoding" if set. Text mode is
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8120488Z         triggered by setting any of text, encoding, errors or universal_newlines.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8120566Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8120724Z         The other arguments are the same as for the Popen constructor.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8120861Z         """
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8120960Z         if input is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8121061Z             if kwargs.get('stdin') is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8121236Z                 raise ValueError('stdin and input arguments may not both be used.')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8121328Z             kwargs['stdin'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8121406Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8121501Z         if capture_output:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8121671Z             if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8121835Z                 raise ValueError('stdout and stderr arguments may not be used '
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8121998Z                                  'with capture_output.')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8122099Z             kwargs['stdout'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8122187Z             kwargs['stderr'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8122264Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8122389Z         with Popen(*popenargs, **kwargs) as process:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8122468Z             try:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8122636Z                 stdout, stderr = process.communicate(input, timeout=timeout)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8122735Z             except TimeoutExpired as exc:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8122832Z                 process.kill()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8122917Z                 if _mswindows:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8123066Z                     # Windows accumulates the output in a single blocking
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8123205Z                     # read() call run on child threads, with the timeout
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8123347Z                     # being done in a join() on those threads.  communicate()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8123494Z                     # _after_ kill() is required to collect that and add it
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8123587Z                     # to the exception.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8123729Z                     exc.stdout, exc.stderr = process.communicate()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8123807Z                 else:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8123956Z                     # POSIX _communicate already populated the output so
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8124082Z                     # far into the TimeoutExpired exception.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8124172Z                     process.wait()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8124255Z                 raise
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8124419Z             except:  # Including KeyboardInterrupt, communicate handled that.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8124511Z                 process.kill()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8124654Z                 # We don't call process.wait() as .__exit__ does that for us.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8124737Z                 raise
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8124835Z             retcode = process.poll()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8124926Z             if check and retcode:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8125065Z >               raise CalledProcessError(retcode, process.args,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8125181Z                                          output=stdout, stderr=stderr)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8125612Z E               subprocess.CalledProcessError: Command '['$PREFIX/bin/python3.12', '$SRC_DIR/docs/Examples/gpu_coagulation_direct.py']' returned non-zero exit status 2.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8125616Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8126330Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/subprocess.py:571: CalledProcessError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8126524Z __________ test_cpu_dilution_example_executes_exact_public_api_decay ___________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8126528Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8126704Z     def test_cpu_dilution_example_executes_exact_public_api_decay() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8126849Z         """Example decays every concentration domain exactly."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8126963Z >       result = _load_example().run_example()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8127049Z                  ^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8127061Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8127174Z particula/tests/dilution_example_test.py:31: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8127366Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8127522Z particula/tests/dilution_example_test.py:25: in _load_example
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8127625Z     spec.loader.exec_module(module)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8127766Z <frozen importlib._bootstrap_external>:995: in exec_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8127852Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8127988Z <frozen importlib._bootstrap_external>:1132: in get_code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8128072Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8128207Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8128211Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8128397Z self = <_frozen_importlib_external.SourceFileLoader object at 0x7f3ea076a330>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8128573Z path = '$SRC_DIR/docs/Examples/cpu_dilution.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8128578Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8128651Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8128905Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/cpu_dilution.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8128909Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8129066Z <frozen importlib._bootstrap_external>:1190: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8129240Z __________ test_cpu_dilution_example_uses_public_runnable_call_chain ___________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8129244Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8129421Z     def test_cpu_dilution_example_uses_public_runnable_call_chain() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8129696Z         """The executable example teaches the supported public runnable API."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8129947Z >       tree = ast.parse(EXAMPLE_PATH.read_text(encoding="utf-8"))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8130114Z                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8130125Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8130311Z particula/tests/dilution_example_test.py:50: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8130531Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8131376Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/pathlib.py:1027: in read_text
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8131662Z     with self.open(mode='r', encoding=encoding, errors=errors) as f:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8131828Z          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8132045Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8132051Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8132290Z self = PosixPath('$SRC_DIR/docs/Examples/cpu_dilution.py')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8132499Z mode = 'r', buffering = -1, encoding = 'utf-8', errors = None, newline = None
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8132503Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8132642Z     def open(self, mode='r', buffering=-1, encoding=None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8132742Z              errors=None, newline=None):
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8132816Z         """
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8132982Z         Open the file pointed to by this path and return a file object, as
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8133081Z         the built-in open() function does.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8133158Z         """
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8133248Z         if "b" not in mode:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8133363Z             encoding = io.text_encoding(encoding)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8133517Z >       return io.open(self, mode, buffering, encoding, errors, newline)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8133629Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8133881Z E       FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/cpu_dilution.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8133886Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8134504Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/pathlib.py:1013: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8134693Z __________ test_cpu_dilution_example_imports_only_public_dependencies __________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8134696Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8134877Z     def test_cpu_dilution_example_imports_only_public_dependencies() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8135095Z         """The example does not reach into concrete implementation modules."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8135248Z >       tree = ast.parse(EXAMPLE_PATH.read_text(encoding="utf-8"))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8135352Z                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8135356Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8135475Z particula/tests/dilution_example_test.py:70: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8135595Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8136593Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/pathlib.py:1027: in read_text
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8136902Z     with self.open(mode='r', encoding=encoding, errors=errors) as f:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8137082Z          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8137226Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8137231Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8137377Z self = PosixPath('$SRC_DIR/docs/Examples/cpu_dilution.py')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8137543Z mode = 'r', buffering = -1, encoding = 'utf-8', errors = None, newline = None
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8137547Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8137680Z     def open(self, mode='r', buffering=-1, encoding=None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8137777Z              errors=None, newline=None):
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8137860Z         """
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8138022Z         Open the file pointed to by this path and return a file object, as
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8138131Z         the built-in open() function does.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8138209Z         """
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8138297Z         if "b" not in mode:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8138408Z             encoding = io.text_encoding(encoding)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8138565Z >       return io.open(self, mode, buffering, encoding, errors, newline)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8138677Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8138917Z E       FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/cpu_dilution.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8138921Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8139546Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/pathlib.py:1013: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8139733Z ___________ test_cpu_dilution_example_results_are_isolated_snapshots ___________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8139737Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8139915Z     def test_cpu_dilution_example_results_are_isolated_snapshots() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8140100Z         """Fresh calls and all initial/final snapshots own independent arrays."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8140194Z >       example = _load_example()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8140284Z                   ^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8140288Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8140409Z particula/tests/dilution_example_test.py:88: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8140528Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8140684Z particula/tests/dilution_example_test.py:25: in _load_example
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8140779Z     spec.loader.exec_module(module)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8140923Z <frozen importlib._bootstrap_external>:995: in exec_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8140998Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8141136Z <frozen importlib._bootstrap_external>:1132: in get_code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8141219Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8141339Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8141346Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8141539Z self = <_frozen_importlib_external.SourceFileLoader object at 0x7f3ea07894f0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8141652Z path = '$SRC_DIR/docs/Examples/cpu_dilution.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8141656Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8141739Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8142048Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/cpu_dilution.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8142053Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8142206Z <frozen importlib._bootstrap_external>:1190: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8142396Z ________ test_cpu_dilution_example_result_rejects_metadata_reassignment ________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8142400Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8142583Z     def test_cpu_dilution_example_result_rejects_metadata_reassignment() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8142738Z         """Example result keeps its execution metadata immutable."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8142849Z >       result = _load_example().run_example()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8142999Z                  ^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8143004Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8143131Z particula/tests/dilution_example_test.py:110: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8143259Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8143421Z particula/tests/dilution_example_test.py:25: in _load_example
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8143519Z     spec.loader.exec_module(module)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8143671Z <frozen importlib._bootstrap_external>:995: in exec_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8143754Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8143890Z <frozen importlib._bootstrap_external>:1132: in get_code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8143969Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8144094Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8144097Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8144307Z self = <_frozen_importlib_external.SourceFileLoader object at 0x7f3ea07c1160>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8144420Z path = '$SRC_DIR/docs/Examples/cpu_dilution.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8144424Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8144510Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8144768Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/cpu_dilution.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8144772Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8144925Z <frozen importlib._bootstrap_external>:1190: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8145108Z ______________ test_cpu_dilution_example_main_reports_all_domains ______________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8145112Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8145290Z capsys = <_pytest.capture.CaptureFixture object at 0x7f3ea07eb7d0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8145294Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8145478Z     def test_cpu_dilution_example_main_reports_all_domains(capsys) -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8145648Z         """Example command reports metadata and before/after snapshots."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8145740Z >       _load_example().main()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8145825Z         ^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8145829Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8145947Z particula/tests/dilution_example_test.py:117: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8146077Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8146398Z particula/tests/dilution_example_test.py:25: in _load_example
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8146503Z     spec.loader.exec_module(module)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8146651Z <frozen importlib._bootstrap_external>:995: in exec_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8146725Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8146873Z <frozen importlib._bootstrap_external>:1132: in get_code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8146948Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8147082Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8147086Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8147284Z self = <_frozen_importlib_external.SourceFileLoader object at 0x7f3ea07e81a0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8147399Z path = '$SRC_DIR/docs/Examples/cpu_dilution.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8147403Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8147483Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8147729Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/cpu_dilution.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8147732Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8147891Z <frozen importlib._bootstrap_external>:1190: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8148070Z _________ test_import_is_lazy_about_warp_and_concrete_capture_modules __________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8148081Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8148260Z     def test_import_is_lazy_about_warp_and_concrete_capture_modules() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8148501Z         """Keep optional capture and concrete composition out of import time."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8148608Z         sys.modules.pop(MODULE_NAME, None)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8148709Z         before = set(sys.modules)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8148834Z >       example = importlib.import_module(MODULE_NAME)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8148936Z                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8148940Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8149100Z particula/tests/gpu_resident_graph_capture_docs_test.py:45: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8149223Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8149906Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8150224Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8150397Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8150605Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8150718Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8150946Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8151062Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8151333Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8151440Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8151699Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8151833Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8152039Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8152179Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8152398Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8152520Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8152772Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8152898Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8153161Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8153293Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8153501Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8153615Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8153829Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8153942Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8154165Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8154170Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8154423Z name = 'docs', import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8154432Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8154518Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8154660Z E   ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8154670Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8154803Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8154994Z __________ test_force_disabled_path_is_deterministic_and_has_no_setup __________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8154998Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8155185Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea07c1fa0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8155189Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8155357Z     def test_force_disabled_path_is_deterministic_and_has_no_setup(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8155469Z         monkeypatch: pytest.MonkeyPatch,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8155550Z     ) -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8155746Z         """Force disabling avoids Warp loading, fixture creation, and fallback."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8155842Z >       example = _fresh_example()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8155930Z                   ^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8155934Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8156089Z particula/tests/gpu_resident_graph_capture_docs_test.py:58: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8156337Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8156556Z particula/tests/gpu_resident_graph_capture_docs_test.py:38: in _fresh_example
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8156672Z     return importlib.import_module(MODULE_NAME)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8156770Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8157517Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8157677Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8157786Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8157906Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8157988Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8158112Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8158193Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8158338Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8158471Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8158619Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8158700Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8158822Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8158895Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8159022Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8159097Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8159246Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8159319Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8159469Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8159547Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8159663Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8159742Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8159861Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8159940Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8160067Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8160074Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8160235Z name = 'docs', import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8160239Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8160317Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8160433Z E   ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8160437Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8160575Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8160761Z _________ test_force_disabled_subprocess_has_exact_address_free_output _________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8160764Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8160958Z     def test_force_disabled_subprocess_has_exact_address_free_output() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8161117Z         """The unavailable branch exits normally without a device."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8161214Z         environment = os.environ | {
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8161352Z             "PARTICULA_EXAMPLE_FORCE_NO_NATIVE_CAPTURE": "1"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8161427Z         }
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8161618Z         result = subprocess.run(  # noqa: S603 - fixed repository-owned script
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8161720Z             [sys.executable, str(SOURCE)],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8161806Z             cwd=ROOT,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8161896Z             env=environment,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8161986Z             check=False,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8162086Z             capture_output=True,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8162165Z             text=True,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8162254Z             timeout=20,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8162332Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8162440Z >       assert result.returncode == 0
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8162521Z E       assert 2 == 0
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8163264Z E        +  where 2 = CompletedProcess(args=['/home/conda/feedstock_root/build_artifacts/particula_1791096598735/_test_env_placehold_placeho.../particula_1791096598735/test_tmp/docs/Examples/gpu_resident_graph_capture.py': [Errno 2] No such file or directory\n").returncode
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8163269Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8163466Z particula/tests/gpu_resident_graph_capture_docs_test.py:99: AssertionError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8163660Z ______ test_preflight_handles_only_explicit_unavailable_cases[None-None] _______
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8163664Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8163860Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea08ecec0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8163954Z warp = None, expected = None
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8163957Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8164064Z     @pytest.mark.parametrize(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8164258Z         "warp, expected",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8164335Z         [
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8164427Z             (None, None),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8164505Z             (
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8164603Z                 SimpleNamespace(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8164762Z                     get_devices=lambda: [_NativeDevice("cpu", is_cuda=False)]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8164846Z                 ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8164928Z                 None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8165006Z             ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8165090Z             (
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8165181Z                 SimpleNamespace(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8165392Z                     get_devices=lambda: [_NativeDevice("cuda:1", is_cuda=True)],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8165497Z                     capture_begin=lambda: None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8165604Z                     capture_end=lambda: None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8165680Z                 ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8165767Z                 "cuda:1",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8165851Z             ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8165928Z             (
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8166024Z                 SimpleNamespace(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8166191Z                     get_devices=lambda: [
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8166331Z                         _NativeDevice("cpu", is_cuda=False),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8166450Z                         _NativeDevice("cuda:1", is_cuda=True),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8166571Z                         _NativeDevice("cuda:2", is_cuda=True),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8166655Z                     ],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8166754Z                     capture_begin=lambda: None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8166860Z                     capture_end=lambda: None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8166965Z                     capture_launch=lambda: None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8167047Z                 ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8167128Z                 "cuda:1",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8167209Z             ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8167285Z         ],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8167369Z     )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8167526Z     def test_preflight_handles_only_explicit_unavailable_cases(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8167703Z         monkeypatch: pytest.MonkeyPatch, warp: Any, expected: str | None
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8167788Z     ) -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8168012Z         """Missing Warp, CUDA, or capture callables return the unavailable result."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8168129Z >       example = _fresh_example()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8168210Z                   ^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8168215Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8168383Z particula/tests/gpu_resident_graph_capture_docs_test.py:145: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8168518Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8168717Z particula/tests/gpu_resident_graph_capture_docs_test.py:38: in _fresh_example
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8168838Z     return importlib.import_module(MODULE_NAME)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8168925Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8169610Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8169766Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8169866Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8169994Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8170069Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8170199Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8170274Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8170427Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8170508Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8170654Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8170732Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8170847Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8170964Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8171165Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8171353Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8171595Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8171719Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8171997Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8172108Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8172321Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8172450Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8172664Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8172786Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8173018Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8173098Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8173357Z name = 'docs', import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8173363Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8173481Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8173643Z E   ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8173650Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8173784Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8173990Z ______ test_preflight_handles_only_explicit_unavailable_cases[warp1-None] ______
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8173994Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8174187Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea1beb380>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8174351Z warp = namespace(get_devices=<function <lambda> at 0x7f3ead9f7d80>)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8174442Z expected = None
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8174446Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8174547Z     @pytest.mark.parametrize(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8174651Z         "warp, expected",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8174727Z         [
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8174821Z             (None, None),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8174900Z             (
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8175002Z                 SimpleNamespace(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8175164Z                     get_devices=lambda: [_NativeDevice("cpu", is_cuda=False)]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8175242Z                 ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8175326Z                 None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8175400Z             ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8175481Z             (
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8175576Z                 SimpleNamespace(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8175743Z                     get_devices=lambda: [_NativeDevice("cuda:1", is_cuda=True)],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8175852Z                     capture_begin=lambda: None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8175951Z                     capture_end=lambda: None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8176033Z                 ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8176194Z                 "cuda:1",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8176285Z             ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8176359Z             (
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8176457Z                 SimpleNamespace(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8176551Z                     get_devices=lambda: [
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8176681Z                         _NativeDevice("cpu", is_cuda=False),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8176803Z                         _NativeDevice("cuda:1", is_cuda=True),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8176914Z                         _NativeDevice("cuda:2", is_cuda=True),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8176997Z                     ],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8177101Z                     capture_begin=lambda: None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8177207Z                     capture_end=lambda: None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8177309Z                     capture_launch=lambda: None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8177395Z                 ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8177487Z                 "cuda:1",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8177561Z             ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8177643Z         ],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8177716Z     )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8177871Z     def test_preflight_handles_only_explicit_unavailable_cases(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8178044Z         monkeypatch: pytest.MonkeyPatch, warp: Any, expected: str | None
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8178134Z     ) -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8178328Z         """Missing Warp, CUDA, or capture callables return the unavailable result."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8178423Z >       example = _fresh_example()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8178513Z                   ^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8178517Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8178673Z particula/tests/gpu_resident_graph_capture_docs_test.py:145: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8178865Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8179065Z particula/tests/gpu_resident_graph_capture_docs_test.py:38: in _fresh_example
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8179187Z     return importlib.import_module(MODULE_NAME)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8179280Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8179953Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8180167Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8180267Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8180390Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8180471Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8180591Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8180672Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8180818Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8180899Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8181044Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8181124Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8181238Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8181320Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8181445Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8181518Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8181665Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8181739Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8181890Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8181963Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8182084Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8182157Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8182282Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8182361Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8182490Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8182494Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8182652Z name = 'docs', import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8182656Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8182728Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8182851Z E   ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8182855Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8182994Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8183187Z _____ test_preflight_handles_only_explicit_unavailable_cases[warp2-cuda:1] _____
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8183194Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8183386Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0834410>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8183834Z warp = namespace(get_devices=<function <lambda> at 0x7f3ead9f6e80>, capture_begin=<function <lambda> at 0x7f3eae568720>, capture_end=<function <lambda> at 0x7f3eae5960c0>)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8183926Z expected = 'cuda:1'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8183930Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8184035Z     @pytest.mark.parametrize(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8184121Z         "warp, expected",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8184202Z         [
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8184282Z             (None, None),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8184363Z             (
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8184452Z                 SimpleNamespace(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8184609Z                     get_devices=lambda: [_NativeDevice("cpu", is_cuda=False)]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8184681Z                 ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8184767Z                 None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8184847Z             ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8184918Z             (
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8185016Z                 SimpleNamespace(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8185164Z                     get_devices=lambda: [_NativeDevice("cuda:1", is_cuda=True)],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8185268Z                     capture_begin=lambda: None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8185363Z                     capture_end=lambda: None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8185444Z                 ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8185529Z                 "cuda:1",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8185647Z             ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8185729Z             (
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8185813Z                 SimpleNamespace(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8185912Z                     get_devices=lambda: [
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8186024Z                         _NativeDevice("cpu", is_cuda=False),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8186217Z                         _NativeDevice("cuda:1", is_cuda=True),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8186337Z                         _NativeDevice("cuda:2", is_cuda=True),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8186422Z                     ],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8186525Z                     capture_begin=lambda: None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8186677Z                     capture_end=lambda: None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8186783Z                     capture_launch=lambda: None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8186856Z                 ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8186942Z                 "cuda:1",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8187013Z             ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8187096Z         ],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8187175Z     )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8187322Z     def test_preflight_handles_only_explicit_unavailable_cases(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8187500Z         monkeypatch: pytest.MonkeyPatch, warp: Any, expected: str | None
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8187575Z     ) -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8187770Z         """Missing Warp, CUDA, or capture callables return the unavailable result."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8187861Z >       example = _fresh_example()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8187950Z                   ^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8187954Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8188116Z particula/tests/gpu_resident_graph_capture_docs_test.py:145: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8188239Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8188437Z particula/tests/gpu_resident_graph_capture_docs_test.py:38: in _fresh_example
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8188549Z     return importlib.import_module(MODULE_NAME)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8188642Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8189292Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8189445Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8189548Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8189662Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8189741Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8189859Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8189942Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8190082Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8190161Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8190307Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8190377Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8190494Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8190564Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8190690Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8190761Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8190906Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8190982Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8191121Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8191199Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8191362Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8191479Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8191672Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8191785Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8191999Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8192004Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8192241Z name = 'docs', import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8192246Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8192359Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8192485Z E   ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8192489Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8192682Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8192870Z _____ test_preflight_handles_only_explicit_unavailable_cases[warp3-cuda:1] _____
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8192874Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8193060Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0836c00>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8193625Z warp = namespace(get_devices=<function <lambda> at 0x7f3eae595da0>, capture_begin=<function <lambda> at 0x7f3eada74180>, capture_end=<function <lambda> at 0x7f3eada740e0>, capture_launch=<function <lambda> at 0x7f3eada74040>)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8193710Z expected = 'cuda:1'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8193755Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8193863Z     @pytest.mark.parametrize(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8193948Z         "warp, expected",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8194030Z         [
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8194117Z             (None, None),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8194190Z             (
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8194286Z                 SimpleNamespace(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8194440Z                     get_devices=lambda: [_NativeDevice("cpu", is_cuda=False)]
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8194523Z                 ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8194600Z                 None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8194683Z             ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8194756Z             (
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8194851Z                 SimpleNamespace(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8195011Z                     get_devices=lambda: [_NativeDevice("cuda:1", is_cuda=True)],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8195111Z                     capture_begin=lambda: None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8195214Z                     capture_end=lambda: None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8195288Z                 ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8195377Z                 "cuda:1",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8195450Z             ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8195529Z             (
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8195615Z                 SimpleNamespace(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8195716Z                     get_devices=lambda: [
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8195838Z                         _NativeDevice("cpu", is_cuda=False),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8195952Z                         _NativeDevice("cuda:1", is_cuda=True),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8196069Z                         _NativeDevice("cuda:2", is_cuda=True),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8196244Z                     ],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8196361Z                     capture_begin=lambda: None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8196455Z                     capture_end=lambda: None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8196560Z                     capture_launch=lambda: None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8196642Z                 ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8196720Z                 "cuda:1",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8196801Z             ),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8196875Z         ],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8196966Z     )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8197115Z     def test_preflight_handles_only_explicit_unavailable_cases(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8197296Z         monkeypatch: pytest.MonkeyPatch, warp: Any, expected: str | None
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8197379Z     ) -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8197562Z         """Missing Warp, CUDA, or capture callables return the unavailable result."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8197661Z >       example = _fresh_example()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8197751Z                   ^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8197755Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8197917Z particula/tests/gpu_resident_graph_capture_docs_test.py:145: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8198039Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8198236Z particula/tests/gpu_resident_graph_capture_docs_test.py:38: in _fresh_example
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8198353Z     return importlib.import_module(MODULE_NAME)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8198441Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8199096Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8199243Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8199350Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8199525Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8199602Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8199731Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8199805Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8199953Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8200024Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8200172Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8200242Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8200363Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8200439Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8200610Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8200687Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8200824Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8200900Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8201038Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8201118Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8201230Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8201306Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8201427Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8201497Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8201627Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8201631Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8201776Z name = 'docs', import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8201779Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8201858Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8201977Z E   ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8201984Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8202114Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8202296Z ____________ test_preflight_propagates_unexpected_errors[failure0] _____________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8202300Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8202477Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3eae22d310>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8202583Z failure = RuntimeError('loader failure')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8202590Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8202690Z     @pytest.mark.parametrize(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8202768Z         "failure",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8202935Z         [RuntimeError("loader failure"), RuntimeError("device failure")],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8203009Z     )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8203137Z     def test_preflight_propagates_unexpected_errors(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8203275Z         monkeypatch: pytest.MonkeyPatch, failure: RuntimeError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8203357Z     ) -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8203526Z         """Unexpected optional-runtime failures do not become fallbacks."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8203627Z >       example = _fresh_example()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8203712Z                   ^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8203716Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8203868Z particula/tests/gpu_resident_graph_capture_docs_test.py:165: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8203997Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8204186Z particula/tests/gpu_resident_graph_capture_docs_test.py:38: in _fresh_example
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8204302Z     return importlib.import_module(MODULE_NAME)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8204394Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8205038Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8205187Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8205285Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8205408Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8205479Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8205604Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8205683Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8205823Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8205902Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8206086Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8206250Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8206372Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8206449Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8206572Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8206644Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8206788Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8206858Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8207004Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8207074Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8207252Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8207322Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8207446Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8207524Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8207648Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8207651Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8207806Z name = 'docs', import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8207810Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8207880Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8207998Z E   ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8208002Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8208135Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8208309Z ____________ test_preflight_propagates_unexpected_errors[failure1] _____________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8208313Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8208497Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0836d80>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8208597Z failure = RuntimeError('device failure')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8208603Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8208706Z     @pytest.mark.parametrize(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8208782Z         "failure",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8208949Z         [RuntimeError("loader failure"), RuntimeError("device failure")],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8209027Z     )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8209150Z     def test_preflight_propagates_unexpected_errors(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8209299Z         monkeypatch: pytest.MonkeyPatch, failure: RuntimeError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8209376Z     ) -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8209553Z         """Unexpected optional-runtime failures do not become fallbacks."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8209642Z >       example = _fresh_example()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8209727Z                   ^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8209731Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8209889Z particula/tests/gpu_resident_graph_capture_docs_test.py:165: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8210061Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8210372Z particula/tests/gpu_resident_graph_capture_docs_test.py:38: in _fresh_example
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8210556Z     return importlib.import_module(MODULE_NAME)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8210701Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8211892Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8212157Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8212335Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8212502Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8212585Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8212797Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8212925Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8213149Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8213258Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8213515Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8213621Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8213826Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8213940Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8214111Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8214182Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8214395Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8214475Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8214616Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8214696Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8214806Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8214882Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8214995Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8215074Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8215205Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8215273Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8215422Z name = 'docs', import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8215427Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8215503Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8215618Z E   ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8215621Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8215760Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8215944Z _________ test_source_contract_preserves_lazy_native_capture_lifecycle _________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8215955Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8216246Z     def test_source_contract_preserves_lazy_native_capture_lifecycle() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8216428Z         """Check import policy and replay ordering in the example source."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8216541Z >       source = SOURCE.read_text(encoding="utf-8")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8216638Z                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8216642Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8216792Z particula/tests/gpu_resident_graph_capture_docs_test.py:177: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8216927Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8217531Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/pathlib.py:1027: in read_text
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8217692Z     with self.open(mode='r', encoding=encoding, errors=errors) as f:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8217801Z          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8217922Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8217925Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8218105Z self = PosixPath('$SRC_DIR/docs/Examples/gpu_resident_graph_capture.py')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8218285Z mode = 'r', buffering = -1, encoding = 'utf-8', errors = None, newline = None
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8218290Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8218421Z     def open(self, mode='r', buffering=-1, encoding=None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8218527Z              errors=None, newline=None):
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8218600Z         """
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8218771Z         Open the file pointed to by this path and return a file object, as
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8218877Z         the built-in open() function does.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8218949Z         """
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8219040Z         if "b" not in mode:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8219150Z             encoding = io.text_encoding(encoding)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8219309Z >       return io.open(self, mode, buffering, encoding, errors, newline)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8219411Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8219711Z E       FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/gpu_resident_graph_capture.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8219715Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8220343Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/pathlib.py:1013: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8220519Z ____________ test_enabled_lifecycle_retires_and_closes_without_cuda ____________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8220523Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8220707Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea08edd90>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8220711Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8220953Z     def test_enabled_lifecycle_retires_and_closes_without_cuda(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8221060Z         monkeypatch: pytest.MonkeyPatch,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8221144Z     ) -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8221332Z         """Exercise the documented capture lifecycle and deterministic teardown."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8221429Z >       example = _fresh_example()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8221511Z                   ^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8221515Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8221672Z particula/tests/gpu_resident_graph_capture_docs_test.py:217: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8221797Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8222048Z particula/tests/gpu_resident_graph_capture_docs_test.py:38: in _fresh_example
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8222171Z     return importlib.import_module(MODULE_NAME)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8222260Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8222922Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8223065Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8223170Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8223293Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8223366Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8223492Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8223565Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8223713Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8223787Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8223936Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8224015Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8224127Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8224203Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8224320Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8224399Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8224539Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8224617Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8224755Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8224832Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8224951Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8225019Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8225140Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8225208Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8225336Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8225343Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8225490Z name = 'docs', import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8225502Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8225571Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8225687Z E   ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8225690Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8225820Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8226006Z __________ test_native_adapter_aborts_and_releases_post_begin_failure __________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8226009Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8226263Z     def test_native_adapter_aborts_and_releases_post_begin_failure() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8226448Z         """End and release an incomplete native capture through the adapter."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8226541Z >       example = _fresh_example()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8226619Z                   ^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8226624Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8226779Z particula/tests/gpu_resident_graph_capture_docs_test.py:400: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8226905Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8227097Z particula/tests/gpu_resident_graph_capture_docs_test.py:38: in _fresh_example
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8227205Z     return importlib.import_module(MODULE_NAME)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8227297Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8228010Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8228153Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8228257Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8228369Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8228448Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8228573Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8228696Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8228847Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8228917Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8229066Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8229136Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8229256Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8229329Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8229456Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8229536Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8229675Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8229755Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8229896Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8229974Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8230084Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8230164Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8230286Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8230356Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8230489Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8230493Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8230635Z name = 'docs', import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8230639Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8230715Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8230826Z E   ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8230838Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8230972Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8231153Z ____________ test_native_adapter_rejects_unsupported_handle_release ____________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8231157Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8231323Z     def test_native_adapter_rejects_unsupported_handle_release() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8231506Z         """Never silently retain a native graph lacking destruction support."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8231594Z >       example = _fresh_example()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8231681Z                   ^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8231684Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8231843Z particula/tests/gpu_resident_graph_capture_docs_test.py:436: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8231967Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8232158Z particula/tests/gpu_resident_graph_capture_docs_test.py:38: in _fresh_example
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8232264Z     return importlib.import_module(MODULE_NAME)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8232360Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8232996Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8233138Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8233240Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8233351Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8233431Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8233553Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8233633Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8233782Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8233853Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8233999Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8234070Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8234232Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8234305Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8234430Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8234500Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8234646Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8234724Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8234864Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8234942Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8235050Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8235126Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8235242Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8235361Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8235483Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8235493Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8235636Z name = 'docs', import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8235640Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8235718Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8235829Z E   ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8235833Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8235970Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8236233Z ______________ test_native_adapter_requires_exact_device_identity ______________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8236246Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8236413Z     def test_native_adapter_requires_exact_device_identity() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8236602Z         """Reject an equal-looking device that is not the selected declaration."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8236692Z >       example = _fresh_example()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8236778Z                   ^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8236786Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8236936Z particula/tests/gpu_resident_graph_capture_docs_test.py:458: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8237064Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8237255Z particula/tests/gpu_resident_graph_capture_docs_test.py:38: in _fresh_example
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8237365Z     return importlib.import_module(MODULE_NAME)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8237459Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8238099Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8238247Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8238352Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8238465Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8238548Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8238669Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8238749Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8238889Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8238969Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8239107Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8239188Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8239308Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8239380Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8239505Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8239575Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8239719Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8239791Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8239936Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8240014Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8240124Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8240209Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8240325Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8240403Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8240525Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8240529Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8240678Z name = 'docs', import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8240738Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8240820Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8240932Z E   ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8240936Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8241072Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8241245Z ______________ test_capability_resolution_rejects_device_mismatch ______________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8241249Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8241414Z     def test_capability_resolution_rejects_device_mismatch() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8241598Z         """Reject a canonical resolver result bound to another device identity."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8241757Z >       example = _fresh_example()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8241840Z                   ^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8241844Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8241995Z particula/tests/gpu_resident_graph_capture_docs_test.py:469: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8242121Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8242311Z particula/tests/gpu_resident_graph_capture_docs_test.py:38: in _fresh_example
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8242428Z     return importlib.import_module(MODULE_NAME)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8242519Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8243154Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8243312Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8243410Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8243532Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8243612Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8243729Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8243806Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8243945Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8244025Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8244167Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8244249Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8244361Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8244440Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8244563Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8244633Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8244777Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8244849Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8244997Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8245070Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8245190Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8245260Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8245384Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8245464Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8245586Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8245590Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8245742Z name = 'docs', import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8245746Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8245815Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8245933Z E   ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8245937Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8246070Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8246329Z ________ test_teardown_attempts_session_close_after_graph_close_failure ________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8246336Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8246536Z     def test_teardown_attempts_session_close_after_graph_close_failure() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8246720Z         """Attempt both teardown operations and chain a second teardown failure."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8246820Z >       example = _fresh_example()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8246900Z                   ^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8246910Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8247059Z particula/tests/gpu_resident_graph_capture_docs_test.py:483: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8247245Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8247439Z particula/tests/gpu_resident_graph_capture_docs_test.py:38: in _fresh_example
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8247553Z     return importlib.import_module(MODULE_NAME)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8247638Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8248282Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8248426Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8248576Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8248695Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8248768Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8248893Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8248963Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8249114Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8249192Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8249332Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8249410Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8249522Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8249600Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8249717Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8249797Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8249943Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8250013Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8250163Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8250235Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8250355Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8250426Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8250551Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8250623Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8250756Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8250760Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8250910Z name = 'docs', import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8250914Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8250986Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8251104Z E   ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8251108Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8251237Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8251549Z __________ test_operation_failure_remains_primary_when_teardown_fails __________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8251558Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8251869Z monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3ea0461e50>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8251880Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8252119Z     def test_operation_failure_remains_primary_when_teardown_fails(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8252288Z         monkeypatch: pytest.MonkeyPatch,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8252402Z     ) -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8252846Z         """Chain cleanup failure without replacing the enabled-path failure."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8253025Z >       example = _fresh_example()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8253164Z                   ^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8253170Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8253446Z particula/tests/gpu_resident_graph_capture_docs_test.py:512: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8253650Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8253857Z particula/tests/gpu_resident_graph_capture_docs_test.py:38: in _fresh_example
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8254073Z     return importlib.import_module(MODULE_NAME)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8254220Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8255457Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/importlib/__init__.py:90: in import_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8255697Z     return _bootstrap._gcd_import(name[level:], package, level)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8255933Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8256234Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8256345Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8256474Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8256547Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8256698Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8256769Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8256919Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8256991Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8257111Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8257246Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8257378Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8257458Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8257599Z <frozen importlib._bootstrap>:1310: in _find_and_load_unlocked
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8257678Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8257819Z <frozen importlib._bootstrap>:488: in _call_with_frames_removed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8257905Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8258017Z <frozen importlib._bootstrap>:1387: in _gcd_import
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8258098Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8258223Z <frozen importlib._bootstrap>:1360: in _find_and_load
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8258293Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8258424Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8258428Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8258572Z name = 'docs', import_ = <function _gcd_import at 0x7f3ee165c0e0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8258575Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8258655Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8258766Z E   ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8258773Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8258907Z <frozen importlib._bootstrap>:1324: ModuleNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8259082Z _________________ test_forced_disabled_script_has_exact_output _________________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8259085Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8259223Z     def test_forced_disabled_script_has_exact_output() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8259401Z         """The standalone disabled script prints deterministic guidance."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8259504Z >       result = subprocess.run(  # noqa: S603
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8259612Z             [sys.executable, str(_EXAMPLE)],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8259699Z             check=True,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8259850Z             capture_output=True,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8260026Z             text=True,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8260267Z             env={**os.environ, "PARTICULA_EXAMPLE_FORCE_NO_WARP": "1"},
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8260395Z             timeout=10,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8260507Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8260514Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8260778Z particula/tests/gpu_resident_multi_timestep_docs_test.py:135: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8260981Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8260995Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8261228Z input = None, capture_output = True, timeout = 10, check = True
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8262440Z popenargs = (['/home/conda/feedstock_root/build_artifacts/particula_1791096598735/_test_env_placehold_placehold_placehold_placehol.../conda/feedstock_root/build_artifacts/particula_1791096598735/test_tmp/docs/Examples/gpu_resident_multi_timestep.py'],)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8263241Z kwargs = {'text': True, 'env': {'DISTRO_ARCH': 'amd64', 'UPLOAD_PACKAGES': 'true', 'CONDA_LIBMAMBA_SOLVER_NO_CHANNELS_FROM_INSTALLED': '1', 'BINSTAR_TOKEN': '', ...}, 'stdout': -1, 'stderr': -1}
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8263248Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8263384Z     def run(*popenargs,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8263692Z             input=None, capture_output=False, timeout=None, check=False, **kwargs):
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8263971Z         """Run command with arguments and return a CompletedProcess instance.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8264089Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8264330Z         The returned instance will have attributes args, returncode, stdout and
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8264531Z         stderr. By default, stdout and stderr are not captured, and those attributes
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8264779Z         will be None. Pass stdout=PIPE and/or stderr=PIPE in order to capture them,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8264908Z         or pass capture_output=True to capture both.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8264989Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8265135Z         If check is True and the exit code was non-zero, it raises a
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8265342Z         CalledProcessError. The CalledProcessError object will have the return code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8265528Z         in the returncode attribute, and output & stderr attributes if those streams
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8265618Z         were captured.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8265689Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8265889Z         If timeout (seconds) is given and the process takes too long,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8266010Z          a TimeoutExpired exception will be raised.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8266217Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8266461Z         There is an optional argument "input", allowing you to
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8266697Z         pass bytes or a string to the subprocess's stdin.  If you use this argument
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8266909Z         you may not also use the Popen constructor's "stdin" argument, as
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8267047Z         it will be used internally.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8267128Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8267314Z         By default, all communication is in bytes, and therefore any "input" should
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8267477Z         be bytes, and the stdout and stderr will be bytes. If in text mode, any
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8267655Z         "input" should be a string, and stdout and stderr will be strings decoded
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8267857Z         according to locale encoding, or by "encoding" if set. Text mode is
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8268051Z         triggered by setting any of text, encoding, errors or universal_newlines.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8268122Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8268281Z         The other arguments are the same as for the Popen constructor.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8268362Z         """
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8268451Z         if input is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8268563Z             if kwargs.get('stdin') is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8268730Z                 raise ValueError('stdin and input arguments may not both be used.')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8268829Z             kwargs['stdin'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8268899Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8268994Z         if capture_output:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8269173Z             if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8269331Z                 raise ValueError('stdout and stderr arguments may not be used '
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8269449Z                                  'with capture_output.')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8269542Z             kwargs['stdout'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8269638Z             kwargs['stderr'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8269710Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8269837Z         with Popen(*popenargs, **kwargs) as process:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8269922Z             try:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8270083Z                 stdout, stderr = process.communicate(input, timeout=timeout)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8270192Z             except TimeoutExpired as exc:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8270282Z                 process.kill()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8270375Z                 if _mswindows:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8270516Z                     # Windows accumulates the output in a single blocking
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8270654Z                     # read() call run on child threads, with the timeout
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8270806Z                     # being done in a join() on those threads.  communicate()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8270942Z                     # _after_ kill() is required to collect that and add it
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8271041Z                     # to the exception.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8271179Z                     exc.stdout, exc.stderr = process.communicate()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8271265Z                 else:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8271401Z                     # POSIX _communicate already populated the output so
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8271527Z                     # far into the TimeoutExpired exception.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8271739Z                     process.wait()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8271818Z                 raise
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8271989Z             except:  # Including KeyboardInterrupt, communicate handled that.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8272076Z                 process.kill()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8272229Z                 # We don't call process.wait() as .__exit__ does that for us.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8272304Z                 raise
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8272404Z             retcode = process.poll()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8272498Z             if check and retcode:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8272627Z >               raise CalledProcessError(retcode, process.args,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8272854Z                                          output=stdout, stderr=stderr)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8273292Z E               subprocess.CalledProcessError: Command '['$PREFIX/bin/python3.12', '$SRC_DIR/docs/Examples/gpu_resident_multi_timestep.py']' returned non-zero exit status 2.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8273297Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8273937Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/subprocess.py:571: CalledProcessError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8274128Z ________ test_enabled_script_runs_warning_free_without_cuda_requirement ________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8274132Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8274215Z     @pytest.mark.warp
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8274408Z     def test_enabled_script_runs_warning_free_without_cuda_requirement() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8274583Z         """The enabled subprocess path is CPU-only and project-warning clean."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8274688Z         pytest.importorskip("warp")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8274793Z >       result = subprocess.run(  # noqa: S603
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8274872Z             [
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8274963Z                 sys.executable,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8275045Z                 "-Werror",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8275241Z                 "-Wignore:Implicitly cleaning up <TemporaryDirectory:ResourceWarning",
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8275330Z                 str(_EXAMPLE),
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8275412Z             ],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8275491Z             check=True,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8275585Z             capture_output=True,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8275669Z             text=True,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8275746Z             env={
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8275829Z                 k: v
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8275931Z                 for k, v in os.environ.items()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8276056Z                 if k != "PARTICULA_EXAMPLE_FORCE_NO_WARP"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8276240Z             },
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8276343Z             timeout=30,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8276418Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8276422Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8276591Z particula/tests/gpu_resident_multi_timestep_docs_test.py:150: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8276724Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8276728Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8276877Z input = None, capture_output = True, timeout = 30, check = True
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8277543Z popenargs = (['/home/conda/feedstock_root/build_artifacts/particula_1791096598735/_test_env_placehold_placehold_placehold_placehol.../conda/feedstock_root/build_artifacts/particula_1791096598735/test_tmp/docs/Examples/gpu_resident_multi_timestep.py'],)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8278001Z kwargs = {'text': True, 'env': {'DISTRO_ARCH': 'amd64', 'UPLOAD_PACKAGES': 'true', 'CONDA_LIBMAMBA_SOLVER_NO_CHANNELS_FROM_INSTALLED': '1', 'BINSTAR_TOKEN': '', ...}, 'stdout': -1, 'stderr': -1}
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8278005Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8278091Z     def run(*popenargs,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8278282Z             input=None, capture_output=False, timeout=None, check=False, **kwargs):
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8278449Z         """Run command with arguments and return a CompletedProcess instance.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8278527Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8278703Z         The returned instance will have attributes args, returncode, stdout and
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8278953Z         stderr. By default, stdout and stderr are not captured, and those attributes
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8279141Z         will be None. Pass stdout=PIPE and/or stderr=PIPE in order to capture them,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8279259Z         or pass capture_output=True to capture both.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8279337Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8279481Z         If check is True and the exit code was non-zero, it raises a
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8279675Z         CalledProcessError. The CalledProcessError object will have the return code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8279857Z         in the returncode attribute, and output & stderr attributes if those streams
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8279994Z         were captured.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8280070Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8280220Z         If timeout (seconds) is given and the process takes too long,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8280341Z          a TimeoutExpired exception will be raised.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8280412Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8280553Z         There is an optional argument "input", allowing you to
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8280729Z         pass bytes or a string to the subprocess's stdin.  If you use this argument
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8280897Z         you may not also use the Popen constructor's "stdin" argument, as
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8280996Z         it will be used internally.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8281067Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8281256Z         By default, all communication is in bytes, and therefore any "input" should
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8281422Z         be bytes, and the stdout and stderr will be bytes. If in text mode, any
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8281599Z         "input" should be a string, and stdout and stderr will be strings decoded
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8281767Z         according to locale encoding, or by "encoding" if set. Text mode is
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8281950Z         triggered by setting any of text, encoding, errors or universal_newlines.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8282029Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8282180Z         The other arguments are the same as for the Popen constructor.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8282262Z         """
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8282354Z         if input is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8282468Z             if kwargs.get('stdin') is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8282635Z                 raise ValueError('stdin and input arguments may not both be used.')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8282734Z             kwargs['stdin'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8282812Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8282899Z         if capture_output:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8283082Z             if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8283239Z                 raise ValueError('stdout and stderr arguments may not be used '
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8283358Z                                  'with capture_output.')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8283446Z             kwargs['stdout'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8283542Z             kwargs['stderr'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8283620Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8283737Z         with Popen(*popenargs, **kwargs) as process:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8283818Z             try:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8283982Z                 stdout, stderr = process.communicate(input, timeout=timeout)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8284087Z             except TimeoutExpired as exc:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8284173Z                 process.kill()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8284264Z                 if _mswindows:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8284401Z                     # Windows accumulates the output in a single blocking
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8284539Z                     # read() call run on child threads, with the timeout
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8284689Z                     # being done in a join() on those threads.  communicate()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8284828Z                     # _after_ kill() is required to collect that and add it
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8284927Z                     # to the exception.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8285064Z                     exc.stdout, exc.stderr = process.communicate()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8285148Z                 else:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8285290Z                     # POSIX _communicate already populated the output so
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8285465Z                     # far into the TimeoutExpired exception.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8285563Z                     process.wait()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8285640Z                 raise
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8285809Z             except:  # Including KeyboardInterrupt, communicate handled that.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8285895Z                 process.kill()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8286048Z                 # We don't call process.wait() as .__exit__ does that for us.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8286190Z                 raise
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8286300Z             retcode = process.poll()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8286398Z             if check and retcode:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8286590Z >               raise CalledProcessError(retcode, process.args,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8286718Z                                          output=stdout, stderr=stderr)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8287385Z E               subprocess.CalledProcessError: Command '['$PREFIX/bin/python3.12', '-Werror', '-Wignore:Implicitly cleaning up <TemporaryDirectory:ResourceWarning', '$SRC_DIR/docs/Examples/gpu_resident_multi_timestep.py']' returned non-zero exit status 2.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8287390Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8288024Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/subprocess.py:571: CalledProcessError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8288215Z _________ test_optimized_enabled_script_keeps_availability_validation __________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8288218Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8288301Z     @pytest.mark.warp
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8288495Z     def test_optimized_enabled_script_keeps_availability_validation() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8288662Z         """Optimized execution retains explicit availability validation."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8288768Z         pytest.importorskip("warp")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8288881Z >       result = subprocess.run(  # noqa: S603
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8288989Z             [sys.executable, "-O", str(_EXAMPLE)],
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8289079Z             check=True,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8289168Z             capture_output=True,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8289252Z             text=True,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8289328Z             env={
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8289419Z                 key: value
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8289537Z                 for key, value in os.environ.items()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8289659Z                 if key != "PARTICULA_EXAMPLE_FORCE_NO_WARP"
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8289739Z             },
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8289819Z             timeout=30,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8289899Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8289902Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8290060Z particula/tests/gpu_resident_multi_timestep_docs_test.py:175: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8290199Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8290202Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8290356Z input = None, capture_output = True, timeout = 30, check = True
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8291015Z popenargs = (['/home/conda/feedstock_root/build_artifacts/particula_1791096598735/_test_env_placehold_placehold_placehold_placehol.../conda/feedstock_root/build_artifacts/particula_1791096598735/test_tmp/docs/Examples/gpu_resident_multi_timestep.py'],)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8291471Z kwargs = {'text': True, 'env': {'DISTRO_ARCH': 'amd64', 'UPLOAD_PACKAGES': 'true', 'CONDA_LIBMAMBA_SOLVER_NO_CHANNELS_FROM_INSTALLED': '1', 'BINSTAR_TOKEN': '', ...}, 'stdout': -1, 'stderr': -1}
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8291476Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8291560Z     def run(*popenargs,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8291743Z             input=None, capture_output=False, timeout=None, check=False, **kwargs):
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8291916Z         """Run command with arguments and return a CompletedProcess instance.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8291987Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8292168Z         The returned instance will have attributes args, returncode, stdout and
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8292347Z         stderr. By default, stdout and stderr are not captured, and those attributes
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8292581Z         will be None. Pass stdout=PIPE and/or stderr=PIPE in order to capture them,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8292704Z         or pass capture_output=True to capture both.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8292786Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8292938Z         If check is True and the exit code was non-zero, it raises a
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8293128Z         CalledProcessError. The CalledProcessError object will have the return code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8293319Z         in the returncode attribute, and output & stderr attributes if those streams
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8293399Z         were captured.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8293478Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8293677Z         If timeout (seconds) is given and the process takes too long,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8293790Z          a TimeoutExpired exception will be raised.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8293868Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8294001Z         There is an optional argument "input", allowing you to
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8294183Z         pass bytes or a string to the subprocess's stdin.  If you use this argument
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8294344Z         you may not also use the Popen constructor's "stdin" argument, as
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8294442Z         it will be used internally.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8294514Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8294697Z         By default, all communication is in bytes, and therefore any "input" should
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8294872Z         be bytes, and the stdout and stderr will be bytes. If in text mode, any
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8295042Z         "input" should be a string, and stdout and stderr will be strings decoded
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8295213Z         according to locale encoding, or by "encoding" if set. Text mode is
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8295390Z         triggered by setting any of text, encoding, errors or universal_newlines.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8295469Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8295627Z         The other arguments are the same as for the Popen constructor.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8295702Z         """
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8295795Z         if input is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8295901Z             if kwargs.get('stdin') is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8296075Z                 raise ValueError('stdin and input arguments may not both be used.')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8296259Z             kwargs['stdin'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8296343Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8296430Z         if capture_output:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8296609Z             if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8296772Z                 raise ValueError('stdout and stderr arguments may not be used '
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8296879Z                                  'with capture_output.')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8296978Z             kwargs['stdout'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8297065Z             kwargs['stderr'] = PIPE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8297144Z     
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8297259Z         with Popen(*popenargs, **kwargs) as process:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8297344Z             try:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8297510Z                 stdout, stderr = process.communicate(input, timeout=timeout)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8297611Z             except TimeoutExpired as exc:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8297706Z                 process.kill()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8297789Z                 if _mswindows:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8297932Z                     # Windows accumulates the output in a single blocking
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8298060Z                     # read() call run on child threads, with the timeout
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8298213Z                     # being done in a join() on those threads.  communicate()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8298356Z                     # _after_ kill() is required to collect that and add it
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8298445Z                     # to the exception.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8298590Z                     exc.stdout, exc.stderr = process.communicate()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8298666Z                 else:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8298807Z                     # POSIX _communicate already populated the output so
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8298923Z                     # far into the TimeoutExpired exception.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8299073Z                     process.wait()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8299160Z                 raise
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8299328Z             except:  # Including KeyboardInterrupt, communicate handled that.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8299429Z                 process.kill()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8299579Z                 # We don't call process.wait() as .__exit__ does that for us.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8299665Z                 raise
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8299761Z             retcode = process.poll()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8299861Z             if check and retcode:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8300001Z >               raise CalledProcessError(retcode, process.args,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8300166Z                                          output=stdout, stderr=stderr)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8300616Z E               subprocess.CalledProcessError: Command '['$PREFIX/bin/python3.12', '-O', '$SRC_DIR/docs/Examples/gpu_resident_multi_timestep.py']' returned non-zero exit status 2.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8300620Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8301253Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/subprocess.py:571: CalledProcessError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8301445Z ________ test_real_warp_cpu_example_has_resident_lifecycle_observations ________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8301448Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8301538Z     @pytest.mark.warp
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8301719Z     def test_real_warp_cpu_example_has_resident_lifecycle_observations() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8301893Z         """The enabled example performs its documented multi-step lifecycle."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8301991Z         pytest.importorskip("warp")
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8302124Z         spec = importlib.util.spec_from_file_location(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8302229Z             "resident_multistep_real", _EXAMPLE
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8302310Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8302446Z         assert spec is not None and spec.loader is not None
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8302568Z         module = importlib.util.module_from_spec(spec)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8302671Z         sys.modules[spec.name] = module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8302746Z         try:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8302851Z >           spec.loader.exec_module(module)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8302855Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8303004Z particula/tests/gpu_resident_multi_timestep_docs_test.py:392: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8303136Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8303278Z <frozen importlib._bootstrap_external>:995: in exec_module
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8303352Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8303489Z <frozen importlib._bootstrap_external>:1132: in get_code
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8303565Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8303696Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8303700Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8303896Z self = <_frozen_importlib_external.SourceFileLoader object at 0x7f3ea04735c0>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8304038Z path = '$SRC_DIR/docs/Examples/gpu_resident_multi_timestep.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8304044Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8304123Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8304407Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/gpu_resident_multi_timestep.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8304411Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8304564Z <frozen importlib._bootstrap_external>:1190: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8304748Z ____ test_cpu_nucleation_example_uses_public_imports_without_source_helpers ____
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8304759Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8304941Z     def test_cpu_nucleation_example_uses_public_imports_without_source_helpers():
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8305138Z         """The runnable example stays on public APIs rather than P2/P3 helpers."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8305272Z >       tree = ast.parse(EXAMPLE.read_text(encoding="utf-8"))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8305379Z                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8305382Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8305499Z particula/tests/nucleation_example_test.py:20: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8305674Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8306374Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/pathlib.py:1027: in read_text
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8306536Z     with self.open(mode='r', encoding=encoding, errors=errors) as f:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8306642Z          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8306765Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8306825Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8307009Z self = PosixPath('$SRC_DIR/docs/Examples/Nucleation/cpu_nucleation.py')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8307185Z mode = 'r', buffering = -1, encoding = 'utf-8', errors = None, newline = None
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8307189Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8307317Z     def open(self, mode='r', buffering=-1, encoding=None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8307421Z              errors=None, newline=None):
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8307496Z         """
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8307666Z         Open the file pointed to by this path and return a file object, as
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8307770Z         the built-in open() function does.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8307843Z         """
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8307933Z         if "b" not in mode:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8308038Z             encoding = io.text_encoding(encoding)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8308200Z >       return io.open(self, mode, buffering, encoding, errors, newline)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8308306Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8308596Z E       FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Nucleation/cpu_nucleation.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8308600Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8309234Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/pathlib.py:1013: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8309414Z ________ test_cpu_nucleation_example_uses_public_api_and_conserves_mass ________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8309418Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8309607Z     def test_cpu_nucleation_example_uses_public_api_and_conserves_mass() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8309766Z         """The one-box example transfers gas while conserving total mass."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8309881Z >       namespace = runpy.run_path(str(EXAMPLE))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8309974Z                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8309978Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8310094Z particula/tests/nucleation_example_test.py:60: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8310223Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8310314Z <frozen runpy>:286: in run_path
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8310393Z     ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8310510Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8310522Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8310667Z fname = '$SRC_DIR/docs/Examples/Nucleation/cpu_nucleation.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8310671Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8310749Z >   ???
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8311016Z E   FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Nucleation/cpu_nucleation.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8311021Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8311129Z <frozen runpy>:254: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8311464Z ______________ test_cpu_nucleation_example_main_is_warning_clean _______________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8311483Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8311696Z     def test_cpu_nucleation_example_main_is_warning_clean() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8311859Z         """The command executes successfully with warnings as errors."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8311969Z         completed = _run_cpu_nucleation_example()
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8312107Z >       assert completed.returncode == 0, completed.stderr
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8312559Z E       AssertionError: $PREFIX/bin/python3.12: can't open file '$SRC_DIR/docs/Examples/Nucleation/cpu_nucleation.py': [Errno 2] No such file or directory
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8312650Z E         
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8312738Z E       assert 2 == 0
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8313431Z E        +  where 2 = CompletedProcess(args=['/home/conda/feedstock_root/build_artifacts/particula_1791096598735/_test_env_placehold_placeho...s/particula_1791096598735/test_tmp/docs/Examples/Nucleation/cpu_nucleation.py': [Errno 2] No such file or directory\n").returncode
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8313436Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8313598Z particula/tests/nucleation_example_test.py:101: AssertionError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8313823Z __________ test_pyproject_marker_list_matches_hook_marker_vocabulary ___________
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8313827Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8314008Z     def test_pyproject_marker_list_matches_hook_marker_vocabulary() -> None:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8314191Z         """Static pytest marker config stays aligned with the hook vocabulary."""
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8314301Z >       assert _load_pyproject_markers() == list(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8314397Z                ^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8314508Z             particula_conftest.PYTEST_MARKER_LINES
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8314590Z         )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8314593Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8314722Z particula/tests/pytest_marker_policy_test.py:137: 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8314844Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8315038Z particula/tests/pytest_marker_policy_test.py:105: in _load_pyproject_markers
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8315141Z     with pyproject_path.open("rb") as file:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8315234Z          ^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8315353Z _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8315356Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8315472Z self = PosixPath('$SRC_DIR/pyproject.toml')
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8315651Z mode = 'rb', buffering = -1, encoding = None, errors = None, newline = None
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8315654Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8315782Z     def open(self, mode='r', buffering=-1, encoding=None,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8315881Z              errors=None, newline=None):
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8315955Z         """
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8316244Z         Open the file pointed to by this path and return a file object, as
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8316353Z         the built-in open() function does.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8316434Z         """
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8316528Z         if "b" not in mode:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8316633Z             encoding = io.text_encoding(encoding)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8316793Z >       return io.open(self, mode, buffering, encoding, errors, newline)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8316902Z                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8317115Z E       FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/pyproject.toml'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8317119Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8317754Z ../_test_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placeh/lib/python3.12/pathlib.py:1013: FileNotFoundError
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8317890Z =========================== short test summary info ============================
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8318838Z FAILED particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py::test_condensation_latent_heat_example_runs_as_main_entrypoint - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8319861Z FAILED particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py::test_condensation_latent_heat_example_runs_without_pint - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8320709Z FAILED particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py::test_condensation_latent_heat_run_example_adds_zero_transfer_explanation - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8321475Z FAILED particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py::test_condensation_latent_heat_main_prints_zero_transfer_explanation - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8322340Z FAILED particula/execution/tests/gpu_resident_session_example_test.py::test_forced_disabled_script_has_exact_stdout - subprocess.CalledProcessError: Command '['$PREFIX/bin/python3.12', '$SRC_DIR/docs/Examples/gpu_resident_session.py']' returned non-zero exit status 2.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8323047Z FAILED particula/execution/tests/gpu_resident_session_example_test.py::test_real_warp_cpu_lifecycle_example - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/gpu_resident_session.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8323597Z FAILED particula/gpu/tests/data_containers_example_test.py::test_example_runs_as_main_entrypoint - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8324190Z FAILED particula/gpu/tests/data_containers_example_test.py::test_guide_runs_as_main_entrypoint - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8324921Z FAILED particula/gpu/tests/data_containers_example_test.py::test_example_non_warp_path_reports_cpu_success - subprocess.CalledProcessError: Command '['$PREFIX/bin/python3.12', '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py']' returned non-zero exit status 2.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8325671Z FAILED particula/gpu/tests/data_containers_example_test.py::test_example_warp_path_reports_round_trip_shapes_and_names - subprocess.CalledProcessError: Command '['$PREFIX/bin/python3.12', '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py']' returned non-zero exit status 2.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8326245Z FAILED particula/gpu/tests/gpu_coagulation_direct_example_test.py::test_forced_disabled_routes_never_import_gpu_runtime - ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8327063Z FAILED particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_forced_disabled_script_is_a_successful_no_kernel_subprocess - subprocess.CalledProcessError: Command '['$PREFIX/bin/python3.12', '$SRC_DIR/docs/Examples/gpu_complete_process_sequence.py']' returned non-zero exit status 2.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8327682Z FAILED particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_main_force_no_warp_prints_oracle_completion - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/gpu_condensation_parity_walkthrough.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8328202Z FAILED particula/gpu/tests/gpu_direct_kernels_example_test.py::test_forced_no_warp_import_run_main_and_subprocess_defer_kernels - ModuleNotFoundError: No module named 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8328764Z FAILED particula/gpu/tests/gpu_direct_kernels_example_test.py::test_main_entrypoint_prints_result_output - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/gpu_direct_kernels_quick_start.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8329399Z FAILED particula/gpu/tests/gpu_direct_nucleation_example_test.py::test_direct_example_runs_with_explicit_transfer_and_conservation - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Nucleation/gpu_direct_nucleation.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8330013Z FAILED particula/gpu/tests/gpu_direct_nucleation_example_test.py::test_direct_example_source_uses_only_documented_boundaries - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Nucleation/gpu_direct_nucleation.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8330757Z FAILED particula/gpu/tests/gpu_direct_nucleation_example_test.py::test_documented_direct_example_command_runs - AssertionError: $PREFIX/bin/python3.12: can't open file '$SRC_DIR/docs/Examples/Nucleation/gpu_direct_nucleation.py': [Errno 2] No such file or directory
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8330840Z   
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8330927Z assert 2 == 0
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8331622Z  +  where 2 = CompletedProcess(args=['/home/conda/feedstock_root/build_artifacts/particula_1791096598735/_test_env_placehold_placeho...cula_1791096598735/test_tmp/docs/Examples/Nucleation/gpu_direct_nucleation.py': [Errno 2] No such file or directory\n").returncode
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8332307Z FAILED particula/tests/backend_selected_coagulation_example_test.py::test_example_forced_no_warp_path - subprocess.CalledProcessError: Command '['$PREFIX/bin/python3.12', '$SRC_DIR/docs/Examples/gpu_coagulation_direct.py']' returned non-zero exit status 2.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8332937Z FAILED particula/tests/dilution_example_test.py::test_cpu_dilution_example_executes_exact_public_api_decay - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/cpu_dilution.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8333458Z FAILED particula/tests/dilution_example_test.py::test_cpu_dilution_example_uses_public_runnable_call_chain - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/cpu_dilution.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8333983Z FAILED particula/tests/dilution_example_test.py::test_cpu_dilution_example_imports_only_public_dependencies - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/cpu_dilution.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8334498Z FAILED particula/tests/dilution_example_test.py::test_cpu_dilution_example_results_are_isolated_snapshots - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/cpu_dilution.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8335029Z FAILED particula/tests/dilution_example_test.py::test_cpu_dilution_example_result_rejects_metadata_reassignment - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/cpu_dilution.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8335531Z FAILED particula/tests/dilution_example_test.py::test_cpu_dilution_example_main_reports_all_domains - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/cpu_dilution.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8335972Z FAILED particula/tests/gpu_resident_graph_capture_docs_test.py::test_import_is_lazy_about_warp_and_concrete_capture_modules - ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8336502Z FAILED particula/tests/gpu_resident_graph_capture_docs_test.py::test_force_disabled_path_is_deterministic_and_has_no_setup - ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8336909Z FAILED particula/tests/gpu_resident_graph_capture_docs_test.py::test_force_disabled_subprocess_has_exact_address_free_output - assert 2 == 0
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8337605Z  +  where 2 = CompletedProcess(args=['/home/conda/feedstock_root/build_artifacts/particula_1791096598735/_test_env_placehold_placeho.../particula_1791096598735/test_tmp/docs/Examples/gpu_resident_graph_capture.py': [Errno 2] No such file or directory\n").returncode
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8338081Z FAILED particula/tests/gpu_resident_graph_capture_docs_test.py::test_preflight_handles_only_explicit_unavailable_cases[None-None] - ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8338542Z FAILED particula/tests/gpu_resident_graph_capture_docs_test.py::test_preflight_handles_only_explicit_unavailable_cases[warp1-None] - ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8339000Z FAILED particula/tests/gpu_resident_graph_capture_docs_test.py::test_preflight_handles_only_explicit_unavailable_cases[warp2-cuda:1] - ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8339468Z FAILED particula/tests/gpu_resident_graph_capture_docs_test.py::test_preflight_handles_only_explicit_unavailable_cases[warp3-cuda:1] - ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8339966Z FAILED particula/tests/gpu_resident_graph_capture_docs_test.py::test_preflight_propagates_unexpected_errors[failure0] - ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8340391Z FAILED particula/tests/gpu_resident_graph_capture_docs_test.py::test_preflight_propagates_unexpected_errors[failure1] - ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8341009Z FAILED particula/tests/gpu_resident_graph_capture_docs_test.py::test_source_contract_preserves_lazy_native_capture_lifecycle - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/gpu_resident_graph_capture.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8341433Z FAILED particula/tests/gpu_resident_graph_capture_docs_test.py::test_enabled_lifecycle_retires_and_closes_without_cuda - ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8341909Z FAILED particula/tests/gpu_resident_graph_capture_docs_test.py::test_native_adapter_aborts_and_releases_post_begin_failure - ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8342330Z FAILED particula/tests/gpu_resident_graph_capture_docs_test.py::test_native_adapter_rejects_unsupported_handle_release - ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8342733Z FAILED particula/tests/gpu_resident_graph_capture_docs_test.py::test_native_adapter_requires_exact_device_identity - ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8343145Z FAILED particula/tests/gpu_resident_graph_capture_docs_test.py::test_capability_resolution_rejects_device_mismatch - ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8343583Z FAILED particula/tests/gpu_resident_graph_capture_docs_test.py::test_teardown_attempts_session_close_after_graph_close_failure - ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8344013Z FAILED particula/tests/gpu_resident_graph_capture_docs_test.py::test_operation_failure_remains_primary_when_teardown_fails - ModuleNotFoundError: No module named 'docs'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8344728Z FAILED particula/tests/gpu_resident_multi_timestep_docs_test.py::test_forced_disabled_script_has_exact_output - subprocess.CalledProcessError: Command '['$PREFIX/bin/python3.12', '$SRC_DIR/docs/Examples/gpu_resident_multi_timestep.py']' returned non-zero exit status 2.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8345720Z FAILED particula/tests/gpu_resident_multi_timestep_docs_test.py::test_enabled_script_runs_warning_free_without_cuda_requirement - subprocess.CalledProcessError: Command '['$PREFIX/bin/python3.12', '-Werror', '-Wignore:Implicitly cleaning up <TemporaryDirectory:ResourceWarning', '$SRC_DIR/docs/Examples/gpu_resident_multi_timestep.py']' returned non-zero exit status 2.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8346556Z FAILED particula/tests/gpu_resident_multi_timestep_docs_test.py::test_optimized_enabled_script_keeps_availability_validation - subprocess.CalledProcessError: Command '['$PREFIX/bin/python3.12', '-O', '$SRC_DIR/docs/Examples/gpu_resident_multi_timestep.py']' returned non-zero exit status 2.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8347194Z FAILED particula/tests/gpu_resident_multi_timestep_docs_test.py::test_real_warp_cpu_example_has_resident_lifecycle_observations - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/gpu_resident_multi_timestep.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8347803Z FAILED particula/tests/nucleation_example_test.py::test_cpu_nucleation_example_uses_public_imports_without_source_helpers - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Nucleation/cpu_nucleation.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8348374Z FAILED particula/tests/nucleation_example_test.py::test_cpu_nucleation_example_uses_public_api_and_conserves_mass - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Nucleation/cpu_nucleation.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8349033Z FAILED particula/tests/nucleation_example_test.py::test_cpu_nucleation_example_main_is_warning_clean - AssertionError: $PREFIX/bin/python3.12: can't open file '$SRC_DIR/docs/Examples/Nucleation/cpu_nucleation.py': [Errno 2] No such file or directory
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8349114Z   
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8349193Z assert 2 == 0
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8349938Z  +  where 2 = CompletedProcess(args=['/home/conda/feedstock_root/build_artifacts/particula_1791096598735/_test_env_placehold_placeho...s/particula_1791096598735/test_tmp/docs/Examples/Nucleation/cpu_nucleation.py': [Errno 2] No such file or directory\n").returncode
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8350452Z FAILED particula/tests/pytest_marker_policy_test.py::test_pyproject_marker_list_matches_hook_marker_vocabulary - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/pyproject.toml'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8351229Z ERROR particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py::test_condensation_latent_heat_run_example_returns_finite_structured_results - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8352031Z ERROR particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py::test_condensation_latent_heat_as_float_uses_first_scalar_value - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8352786Z ERROR particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py::test_condensation_latent_heat_build_aerosol_creates_single_box_state - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8353530Z ERROR particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py::test_condensation_latent_heat_main_path_matches_run_example_contract - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8354330Z ERROR particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py::test_condensation_latent_heat_example_reports_condensation_or_explicit_zero_transfer - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8355104Z ERROR particula/dynamics/condensation/tests/condensation_latent_heat_example_test.py::test_condensation_latent_heat_example_energy_matches_mass_transfer_contract - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8355562Z ERROR particula/execution/tests/gpu_resident_session_example_test.py::test_forced_disable_skips_loader_and_fixture - ModuleNotFoundError: No module named 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8355990Z ERROR particula/execution/tests/gpu_resident_session_example_test.py::test_missing_warp_skips_fixture - ModuleNotFoundError: No module named 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8356561Z ERROR particula/execution/tests/gpu_resident_session_example_test.py::test_broken_enabled_warp_import_propagates[error0] - ModuleNotFoundError: No module named 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8357046Z ERROR particula/execution/tests/gpu_resident_session_example_test.py::test_broken_enabled_warp_import_propagates[error1] - ModuleNotFoundError: No module named 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8357542Z ERROR particula/execution/tests/gpu_resident_session_example_test.py::test_loader_orders_concrete_imports_without_gpu_package - ModuleNotFoundError: No module named 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8358063Z ERROR particula/execution/tests/gpu_resident_session_example_test.py::test_enabled_loader_errors_propagate_without_fixture_or_output[error0] - ModuleNotFoundError: No module named 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8358583Z ERROR particula/execution/tests/gpu_resident_session_example_test.py::test_enabled_loader_errors_propagate_without_fixture_or_output[error1] - ModuleNotFoundError: No module named 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8359044Z ERROR particula/execution/tests/gpu_resident_session_example_test.py::test_main_propagates_an_enabled_loader_error - ModuleNotFoundError: No module named 'gpu_resident_session'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8359684Z ERROR particula/gpu/tests/data_containers_example_test.py::test_build_particle_data_returns_documented_shapes - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8360288Z ERROR particula/gpu/tests/data_containers_example_test.py::test_build_gas_data_returns_documented_shapes_and_names - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8360877Z ERROR particula/gpu/tests/data_containers_example_test.py::test_warp_enabled_honors_force_no_warp_environment - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8361547Z ERROR particula/gpu/tests/data_containers_example_test.py::test_run_example_reports_cpu_only_message_when_warp_disabled - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8362185Z ERROR particula/gpu/tests/data_containers_example_test.py::test_run_example_falls_back_to_cpu_message_when_warp_transfer_fails - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8362750Z ERROR particula/gpu/tests/data_containers_example_test.py::test_example_main_prints_example_output - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8363349Z ERROR particula/gpu/tests/data_containers_example_test.py::test_guide_main_prints_example_output - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8363990Z ERROR particula/gpu/tests/data_containers_example_test.py::test_guide_module_re_exports_canonical_run_example - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8364686Z ERROR particula/gpu/tests/data_containers_example_test.py::test_guide_module_raises_import_error_when_canonical_example_cannot_load - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/Data_Containers/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8365297Z ERROR particula/gpu/tests/data_containers_example_test.py::test_run_example_warp_path_reports_round_trip_shapes_and_names - FileNotFoundError: [Errno 2] No such file or directory: '$SRC_DIR/docs/Examples/data_containers_and_gpu_foundations.py'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8365792Z ERROR particula/gpu/tests/gpu_coagulation_direct_example_test.py::test_cpu_fixture_has_documented_active_and_inactive_slots - ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8366334Z ERROR particula/gpu/tests/gpu_coagulation_direct_example_test.py::test_runtime_loader_uses_selected_adapter_imports - ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8366853Z ERROR particula/gpu/tests/gpu_coagulation_direct_example_test.py::test_enabled_path_uses_selected_adapter_and_explicit_lifecycle - ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8367342Z ERROR particula/gpu/tests/gpu_coagulation_direct_example_test.py::test_failures_propagate_without_fallback_or_restore[loader] - ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8367882Z ERROR particula/gpu/tests/gpu_coagulation_direct_example_test.py::test_failures_propagate_without_fallback_or_restore[conversion] - ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8368384Z ERROR particula/gpu/tests/gpu_coagulation_direct_example_test.py::test_failures_propagate_without_fallback_or_restore[first] - ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8368950Z ERROR particula/gpu/tests/gpu_coagulation_direct_example_test.py::test_failures_propagate_without_fallback_or_restore[second] - ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8369457Z ERROR particula/gpu/tests/gpu_coagulation_direct_example_test.py::test_failures_propagate_without_fallback_or_restore[synchronize] - ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8369954Z ERROR particula/gpu/tests/gpu_coagulation_direct_example_test.py::test_failures_propagate_without_fallback_or_restore[restore] - ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8370467Z ERROR particula/gpu/tests/gpu_coagulation_direct_example_test.py::test_real_warp_selected_adapter_reuses_rng_and_preserves_identity - ModuleNotFoundError: No module named 'gpu_coagulation_direct'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8371040Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_build_cpu_state_has_documented_sparse_float64_schema - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8371566Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_forced_disabled_path_does_not_reach_enabled_loader - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8372104Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_warp_enabled_handles_import_failure_and_available_runtime - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8372650Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_runtime_unavailable_returns_metadata_without_device_transfers - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8373212Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_enabled_runtime_loading_failure_propagates_without_success_output - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8373829Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_setup_conversion_failure_does_not_continue_to_sidecars_or_steps[to_warp_particle_data] - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8374429Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_setup_conversion_failure_does_not_continue_to_sidecars_or_steps[to_warp_gas_data] - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8375055Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_setup_conversion_failure_does_not_continue_to_sidecars_or_steps[to_warp_environment_data] - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8375607Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_load_enabled_runtime_defers_and_collects_required_boundaries - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8376227Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_enabled_path_converts_once_orders_steps_and_restores_once - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8376865Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_boundary_failure_propagates_stops_later_calls_and_prevents_early_restore[condensation] - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8377476Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_boundary_failure_propagates_stops_later_calls_and_prevents_early_restore[coagulation] - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8378070Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_boundary_failure_propagates_stops_later_calls_and_prevents_early_restore[dilution] - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8378728Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_boundary_failure_propagates_stops_later_calls_and_prevents_early_restore[wall_loss] - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8379333Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_boundary_failure_propagates_stops_later_calls_and_prevents_early_restore[nucleation] - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8379950Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_boundary_failure_propagates_stops_later_calls_and_prevents_early_restore[synchronize] - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8380635Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_boundary_failure_propagates_stops_later_calls_and_prevents_early_restore[restore_particles] - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8381235Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_boundary_failure_propagates_stops_later_calls_and_prevents_early_restore[restore_gas] - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8381871Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_boundary_failure_propagates_stops_later_calls_and_prevents_early_restore[restore_environment] - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8382339Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_main_prints_only_example_output - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8382844Z ERROR particula/gpu/tests/gpu_complete_process_sequence_example_test.py::test_real_warp_cpu_path_restores_named_cpu_containers - ModuleNotFoundError: No module named 'gpu_complete_process_sequence'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8383392Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_fixture_and_builders_are_fp64_readonly_and_non_aliasing - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8383955Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_oracle_has_four_substep_uptake_evaporation_coupling_and_energy - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8384447Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_oracle_does_not_mutate_supplied_source - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8385103Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_fixture_validation_rejects_nonfinite_and_invalid_physical_values[temperature-value0-finite] - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8385741Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_fixture_validation_rejects_nonfinite_and_invalid_physical_values[pressure-value1-finite] - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8386448Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_fixture_validation_rejects_nonfinite_and_invalid_physical_values[density-value2-positive] - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8387147Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_fixture_validation_rejects_nonfinite_and_invalid_physical_values[gas_concentration-value3-nonnegative] - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8387702Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_fixture_validation_rejects_unsupported_thermodynamic_mode - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8388321Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_disabled_or_unavailable_warp_completes_oracle_without_runtime_work - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8388852Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_force_disabled_warp_defers_runtime_after_oracle - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8389430Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_fake_enabled_route_has_explicit_sidecars_and_synchronized_readback - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8390042Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_oracle_completes_before_runtime_and_ignores_warp_source_mutation - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8390639Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_enabled_failures_propagate_after_completed_oracle_without_restore[loader] - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8391272Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_enabled_failures_propagate_after_completed_oracle_without_restore[particle conversion] - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8391879Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_enabled_failures_propagate_after_completed_oracle_without_restore[gas conversion] - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8392493Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_enabled_failures_propagate_after_completed_oracle_without_restore[allocation] - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8393083Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_enabled_failures_propagate_after_completed_oracle_without_restore[kernel] - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8393613Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_acceptance_categories_are_independently_evaluated - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8394161Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_acceptance_reports_all_categories_after_multiple_failures - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8394649Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_warp_cpu_matches_independent_oracle - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8395177Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_cuda_matches_independent_oracle_when_available - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8395687Z ERROR particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py::test_main_returns_nonzero_for_failed_acceptance - ModuleNotFoundError: No module named 'gpu_condensation_parity_walkthrough'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8396310Z ERROR particula/gpu/tests/gpu_direct_kernels_example_test.py::test_cpu_builders_preserve_documented_dtype_and_species_order - ModuleNotFoundError: No module named 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8396834Z ERROR particula/gpu/tests/gpu_direct_kernels_example_test.py::test_load_gpu_runtime_imports_only_direct_condensation_contract - ModuleNotFoundError: No module named 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8397326Z ERROR particula/gpu/tests/gpu_direct_kernels_example_test.py::test_unavailable_warp_skips_loader_and_conversions - ModuleNotFoundError: No module named 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8397874Z ERROR particula/gpu/tests/gpu_direct_kernels_example_test.py::test_enabled_path_reuses_complete_caller_owned_sidecars - ModuleNotFoundError: No module named 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8398400Z ERROR particula/gpu/tests/gpu_direct_kernels_example_test.py::test_kernel_failure_propagates_without_restore_or_success_output[1] - ModuleNotFoundError: No module named 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8398916Z ERROR particula/gpu/tests/gpu_direct_kernels_example_test.py::test_kernel_failure_propagates_without_restore_or_success_output[2] - ModuleNotFoundError: No module named 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8399449Z ERROR particula/gpu/tests/gpu_direct_kernels_example_test.py::test_real_warp_cpu_path_reuses_sidecars_and_couples_gas - ModuleNotFoundError: No module named 'gpu_direct_kernels_quick_start'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8399909Z ERROR particula/tests/gpu_resident_multi_timestep_docs_test.py::test_forced_disable_runs_no_enabled_work - ModuleNotFoundError: No module named 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8400378Z ERROR particula/tests/gpu_resident_multi_timestep_docs_test.py::test_missing_top_level_warp_runs_no_enabled_work - ModuleNotFoundError: No module named 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8400821Z ERROR particula/tests/gpu_resident_multi_timestep_docs_test.py::test_broken_warp_import_propagates[error0] - ModuleNotFoundError: No module named 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8401271Z ERROR particula/tests/gpu_resident_multi_timestep_docs_test.py::test_broken_warp_import_propagates[error1] - ModuleNotFoundError: No module named 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8401732Z ERROR particula/tests/gpu_resident_multi_timestep_docs_test.py::test_loader_requests_only_concrete_resident_seams - ModuleNotFoundError: No module named 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8402234Z ERROR particula/tests/gpu_resident_multi_timestep_docs_test.py::test_enabled_loader_error_propagates_without_fixture[error0] - ModuleNotFoundError: No module named 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8402729Z ERROR particula/tests/gpu_resident_multi_timestep_docs_test.py::test_enabled_loader_error_propagates_without_fixture[error1] - ModuleNotFoundError: No module named 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8403204Z ERROR particula/tests/gpu_resident_multi_timestep_docs_test.py::test_availability_failure_precedes_fixture_and_setup - ModuleNotFoundError: No module named 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8403704Z ERROR particula/tests/gpu_resident_multi_timestep_docs_test.py::test_setup_failure_propagates_without_checkpoint_or_restart - ModuleNotFoundError: No module named 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8404189Z ERROR particula/tests/gpu_resident_multi_timestep_docs_test.py::test_writer_dispatch_failure_propagates_after_guard_close - ModuleNotFoundError: No module named 'gpu_resident_multi_timestep'
linux_64_	UNKNOWN STEP	2026-10-04T06:57:46.8404387Z = 50 failed, 5800 passed, 166 skipped, 57 deselected, 94 errors in 351.17s (0:05:51) =
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7041723Z WARNING: Tests failed for particula-0.2.14-pyhd8ed1ab_0.conda - moving package to /home/conda/feedstock_root/build_artifacts/broken
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7349330Z Traceback (most recent call last):
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7352821Z   File "/opt/conda/lib/python3.14/site-packages/conda_build/build.py", line 3523, in test
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7353561Z     utils.check_call_env(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7353801Z     ~~~~~~~~~~~~~~~~~~~~^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7354016Z         cmd,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7354200Z         ^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7354386Z     ...<3 lines>...
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7354607Z         rewrite_stdout_env=rewrite_env,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7354900Z         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7355131Z     )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7355305Z     ^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7355652Z   File "/opt/conda/lib/python3.14/site-packages/conda_build/utils.py", line 413, in check_call_env
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7356339Z     return _func_defaulting_env_to_os_environ("call", *popenargs, **kwargs)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7357194Z   File "/opt/conda/lib/python3.14/site-packages/conda_build/utils.py", line 389, in _func_defaulting_env_to_os_environ
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7357764Z     raise subprocess.CalledProcessError(proc.returncode, _args)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7360284Z subprocess.CalledProcessError: Command '['/bin/bash', '-o', 'errexit', '/home/conda/feedstock_root/build_artifacts/particula_1791096598735/test_tmp/conda_test_runner.sh']' returned non-zero exit status 1.
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7361029Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7361214Z During handling of the above exception, another exception occurred:
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7361603Z 
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7361711Z Traceback (most recent call last):
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7362052Z   File "/opt/conda/bin/conda-build", line 10, in <module>
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7362351Z     sys.exit(execute())
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7362564Z              ~~~~~~~^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7362957Z   File "/opt/conda/lib/python3.14/site-packages/conda_build/cli/main_build.py", line 639, in execute
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7363379Z     api.build(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7363581Z     ~~~~~~~~~^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7363780Z         parsed.recipe,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7363987Z         ^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7364187Z     ...<8 lines>...
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7364409Z         cache_dir=parsed.cache_dir,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7364659Z         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7364883Z     )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7365059Z     ^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7365376Z   File "/opt/conda/lib/python3.14/site-packages/conda_build/api.py", line 242, in build
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7365757Z     return build_tree(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7365972Z         recipes,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7366515Z     ...<7 lines>...
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7366743Z         variants=variants,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7366957Z     )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7367296Z   File "/opt/conda/lib/python3.14/site-packages/conda_build/build.py", line 3710, in build_tree
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7367763Z     test(pkg, config=metadata.config.copy(), stats=stats)
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7368094Z     ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7368485Z   File "/opt/conda/lib/python3.14/site-packages/conda_build/build.py", line 3537, in test
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7368861Z     tests_failed(
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7369052Z     ~~~~~~~~~~~~^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7369239Z         metadata,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7369425Z         ^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7369623Z     ...<2 lines>...
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7369831Z         config=metadata.config,
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7370065Z         ^^^^^^^^^^^^^^^^^^^^^^^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7370275Z     )
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7370443Z     ^
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7370782Z   File "/opt/conda/lib/python3.14/site-packages/conda_build/build.py", line 3584, in tests_failed
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7371455Z     raise CondaBuildUserError("TESTS FAILED: " + os.path.basename(pkg))
linux_64_	UNKNOWN STEP	2026-10-04T06:57:50.7371978Z conda_build.exceptions.CondaBuildUserError: TESTS FAILED: particula-0.2.14-pyhd8ed1ab_0.conda
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.0404930Z ##[error]Process completed with exit code 1.
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.0631018Z Post job cleanup.
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.1425212Z [command]/usr/bin/git version
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.1468756Z git version 2.55.0
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.1498948Z Temporarily overriding HOME='/home/runner/work/_temp/e6573427-2c35-4350-a921-056ff4d4961b' before making global git config changes
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.1500027Z Adding repository directory to the temporary git global config as a safe directory
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.1503255Z [command]/usr/bin/git config --global --add safe.directory /home/runner/work/particula-feedstock/particula-feedstock
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.1534639Z Removing SSH command configuration
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.1541352Z [command]/usr/bin/git config --local --name-only --get-regexp core\.sshCommand
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.1572601Z [command]/usr/bin/git submodule foreach --recursive sh -c "git config --local --name-only --get-regexp 'core\.sshCommand' && git config --local --unset-all 'core.sshCommand' || :"
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.1762961Z Removing HTTP extra header
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.1767731Z [command]/usr/bin/git config --local --name-only --get-regexp http\.https\:\/\/github\.com\/\.extraheader
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.1798159Z [command]/usr/bin/git submodule foreach --recursive sh -c "git config --local --name-only --get-regexp 'http\.https\:\/\/github\.com\/\.extraheader' && git config --local --unset-all 'http.https://github.com/.extraheader' || :"
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.1988055Z Removing includeIf entries pointing to credentials config files
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.1993512Z [command]/usr/bin/git config --local --name-only --get-regexp ^includeIf\.gitdir:
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.2020524Z includeif.gitdir:/home/runner/work/particula-feedstock/particula-feedstock/.git.path
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.2021227Z includeif.gitdir:/home/runner/work/particula-feedstock/particula-feedstock/.git/worktrees/*.path
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.2021683Z includeif.gitdir:/github/workspace/.git.path
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.2022020Z includeif.gitdir:/github/workspace/.git/worktrees/*.path
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.2029595Z [command]/usr/bin/git config --local --get-all includeif.gitdir:/home/runner/work/particula-feedstock/particula-feedstock/.git.path
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.2049218Z /home/runner/work/_temp/git-credentials-09aca1b2-0811-4c00-aca8-57a0ad25ac05.config
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.2058085Z [command]/usr/bin/git config --local --unset includeif.gitdir:/home/runner/work/particula-feedstock/particula-feedstock/.git.path \/home\/runner\/work\/_temp\/git\-credentials\-09aca1b2\-0811\-4c00\-aca8\-57a0ad25ac05\.config
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.2090918Z [command]/usr/bin/git config --local --get-all includeif.gitdir:/home/runner/work/particula-feedstock/particula-feedstock/.git/worktrees/*.path
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.2141180Z /home/runner/work/_temp/git-credentials-09aca1b2-0811-4c00-aca8-57a0ad25ac05.config
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.2149318Z [command]/usr/bin/git config --local --unset includeif.gitdir:/home/runner/work/particula-feedstock/particula-feedstock/.git/worktrees/*.path \/home\/runner\/work\/_temp\/git\-credentials\-09aca1b2\-0811\-4c00\-aca8\-57a0ad25ac05\.config
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.2184150Z [command]/usr/bin/git config --local --get-all includeif.gitdir:/github/workspace/.git.path
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.2202383Z /github/runner_temp/git-credentials-09aca1b2-0811-4c00-aca8-57a0ad25ac05.config
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.2208986Z [command]/usr/bin/git config --local --unset includeif.gitdir:/github/workspace/.git.path \/github\/runner_temp\/git\-credentials\-09aca1b2\-0811\-4c00\-aca8\-57a0ad25ac05\.config
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.2238884Z [command]/usr/bin/git config --local --get-all includeif.gitdir:/github/workspace/.git/worktrees/*.path
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.2258009Z /github/runner_temp/git-credentials-09aca1b2-0811-4c00-aca8-57a0ad25ac05.config
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.2265065Z [command]/usr/bin/git config --local --unset includeif.gitdir:/github/workspace/.git/worktrees/*.path \/github\/runner_temp\/git\-credentials\-09aca1b2\-0811\-4c00\-aca8\-57a0ad25ac05\.config
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.2296925Z [command]/usr/bin/git submodule foreach --recursive git config --local --show-origin --name-only --get-regexp remote.origin.url
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.2479034Z Removing credentials config '/home/runner/work/_temp/git-credentials-09aca1b2-0811-4c00-aca8-57a0ad25ac05.config'
linux_64_	UNKNOWN STEP	2026-10-04T06:58:00.2628585Z Cleaning up orphan processes
