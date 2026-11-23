# Milestone 8 — AIOMFAC PyPI release and maintainer runbook

A public release is complete when a clean environment installs the extra from PyPI and reproduces the mixed reference case. The first pair targets Particula 0.3.5 and particula-aiomfac 0.1.4. Subsequent compatible backend versions can ship independently. The base extra accepts backend versions from 0.1.4 up to 0.2. CPU runners qualify the release files on Python 3.12, 3.13 and 3.14. The standalone backend also offers a CUDA extra in the same wheel. Its device qualification is recorded in the [scientific study](../docs/studies/validation/condensation/aiomfac-sparse/index.md).

## Release decision

| Upstream event | Action |
| --- | --- |
| PR, merge queue, manual rehearsal or fork main push | Build both universal wheels from their source archives, retain the four files and installation evidence. Publication is disabled. |
| First backend introduction or stable backend version increase on main | Create the backend GitHub release and publish the verified backend with its dedicated PyPI identity. For a backend-only release, exercise the candidate with the compatible public base before upload and repeat with both packages from PyPI afterward. |
| Base version increase or matching manually published base release tag | Run the existing base, plugin and scientific gates. Install the candidate base extra against public PyPI before publishing. |
| Both versions increase | Qualify the pair, publish the backend, verify both public backend hashes, and then publish the base. |
| No version increase, backend tag or helper tag | No new production Python upload. A downgrade or missing comparison commit fails planning. |

Compare each version to the upstream push's previous commit. A valid commit lacking the backend file establishes the initial release. Backend GitHub tags use `particula-aiomfac-vVERSION` and leave the base release as GitHub's latest.

## Before merging a release

1. Read the target version from each package's literal `__version__`. Compare with [public Particula](https://pypi.org/pypi/particula/json) and [public backend metadata](https://pypi.org/pypi/particula-aiomfac/json). At first setup the base was 0.3.4 and the backend returned HTTP 404. A version that already has different bytes requires a new version.
2. For first setup, a PyPI project owner registers the pending project `particula-aiomfac` for GitHub owner `uncscode`, repository `particula`, workflow filename `pypi.yml`, and environment `pypi-aiomfac`. Restrict this GitHub environment to branch `main`. Both settings were configured during the first rehearsal. PyPI creates the project on the first successful upload. [PyPI pending publisher setup](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
3. Review the PR's CPU matrix and the `python-release-<SHA>` artifact. It contains the base and backend folders, `release.json` with versions, hashes, commit, upstream model revision and parameter digest, a documentation bundle report, and `SHA256SUMS`. Review the model pin and existing license notices when updating parameters or the reference. The PyPI action's backend upload job alone receives OIDC permission. [PyPA publisher](https://github.com/pypa/gh-action-pypi-publish)
4. A maintainer merges the version bump into upstream main when ready to publish. The production run builds the merged commit, qualifies those bytes, archives them and uploads them. A PR rehearsal verifies its own commit.

## Rehearse with the candidate files

From the repository root, with Python 3.12, build and audit each project once.

```sh
out=.artifacts/aiomfac-release
python -m build --outdir "$out/base"
python -m build --outdir "$out/backend" packages/particula-aiomfac
python scripts/check_doc_bundle.py --wheel "$out"/base/*.whl --sdist "$out"/base/*.tar.gz --report "$out/docs-bundle.json"
python packages/particula-aiomfac/tools/qualify_packages.py --from-dist "$out" --audit-only --manifest "$out/release.json"
python packages/particula-aiomfac/tools/qualify_packages.py --from-dist "$out" --sdists
```

Repeat the last command on Python 3.13 and 3.14 without `--sdists`. The matrix installs the base alone, the backend alone and the base with its extra. Installed probes run outside the checkout with Fortran off PATH. The backend builds and reloads a sparse water table and checks neutral and salt results. The combined installation runs native condensation including a mapped hybrid ROS3 trajectory against CPU Radau. The separate reference workflow regenerates scientific fixtures from pinned Fortran.

The base archives retain the MIT package and the published examples. The backend archives carry the GPL license, notices and generated model data. The sdist also provides the reference manifest, regeneration tools and fixture corpus. Upstream MINPACK, the Numerical Recipes-related bracketing file, string utilities and compiler-generated modules are reference-build inputs. The archive audit checks for excluded reference binaries, Fortran source and accidental backend payloads in the base.

## Verify publication and recover a partial run

Check the Actions summary, version-specific public hashes and backend attestation. From a fresh environment the public probe installs the extra using only PyPI, checks its resolved versions and calls the installed adapter. Expected at 298 K are water activity 0.9878567852129456 and mean NaCl molal coefficient 0.7037048661941817.

```sh
python packages/particula-aiomfac/tools/public_probe.py --base-version 0.3.5 --backend-version 0.1.4
```

For a later backend-only release supply just `--backend-version`. Record the public release URLs and successful run in the release notes. GitHub tags, a successful build and configuration of the pending publisher are intermediate milestones.

If a job fails, inspect the per-version registry response and the tagged release assets. Rerun **failed jobs** in the original workflow run to reuse the qualified artifacts, retained for 30 days. Matching public files can be skipped, and the publisher can supply missing files. Hash conflicts, a moved tag or a yanked file stop publication. If the original run artifacts expire, recover the matching archives from the exact GitHub release and verify checksums before arranging a maintainer recovery; otherwise qualify a new version. A joint release can resume the base leg after backend verification. [PyPI release metadata](https://docs.pypi.org/api/json/) · [Immutable PyPI filenames](https://docs.pypi.org/publishing/troubleshooting/)

## First release rehearsal

[Draft PR 1720](https://github.com/uncscode/particula/pull/1720) rehearses the joint release. The [first hosted candidate build and installation matrix](https://github.com/uncscode/particula/actions/runs/45093663350) passed on all three supported Python versions. It archived two wheels and two source distributions under one checksum manifest. The jobs tested the mixed reference, closed box, hybrid ROS3 trajectory and installed table CLI. The local Python 3.12, 3.13 and 3.14 runs also passed, including both source installs on 3.12. The independent GNU 16.2 reference check reproduced all 28 cases from the offline cache.

The initial hosted checks also exposed outdated capability counts, Warp typing errors and a compiler-dependent unsuccessful trace solve. The follow-up adds prepared-provider catalogue coverage, checks the Warp annotations, and records the GNU 13 diagnostics for that one failed solve. The archived CUDA box and parcel figures were redrawn on CPU using the original numerical data and the matching source commit. The [Ubuntu GNU 13 reference run](https://github.com/uncscode/particula/actions/runs/45094858496) then passed fresh and offline verification. In a clean Python 3.12 installation of the local wheels from commit `edd4d713ce`, the measured water activity was 0.9878567852129452 and the NaCl mean molal coefficient was 0.7037048661941815. Production release evidence belongs here after the maintainer merge and successful public installation.
