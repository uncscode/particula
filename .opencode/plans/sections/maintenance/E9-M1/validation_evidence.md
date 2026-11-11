# E9-M1 P5 validation and handoff evidence

## Scope and revision identity

Date: 2026-11-10 (America/Denver). Issue: #1614.

The accumulation branch contains the P1–P4 commits, merge `2c94bdf3d` of
upstream main `8dc8bc51f`, and ignore cherry-pick `c051f27b1` of
`e2ecb1d03`. P5 adds a developer reference, ledger reconciliation and the
four-case data-only integration regression. No production API is added by P5.

Final validation is recorded against an exact committed revision in the PR.
This file is the durable gate summary; later evidence-only commits must identify
which tested revision they describe instead of claiming tests ran on their own
future commit hash. Literal logs are retained locally during validation, and
command/results excerpts are published in the PR for review.

## Environment

Repository `.venv`: Python 3.12.12, Warp 1.11.1. With host device access,
Warp reports both `cpu` and `cuda:0`: NVIDIA GeForce RTX 5060, 8 GiB, sm_120,
CUDA Toolkit 12.9 and driver CUDA compatibility 13.4 (NVIDIA 615.78.08).
The initially restricted probe hid CUDA and is not hardware-absence evidence.
Final GPU-capable runs must use host device access and report actual results;
optional skips are never CUDA success or fallback evidence.

## Current gate status

The default lint failure was fixed in `0db7e42b2`: omit the implicit positional
`.` from mypy so it honors the existing `[tool.mypy] packages = ["particula"]`
configuration. This matches CI without exclusions, diagnostic suppression, or
changes to package type-checking policy. Explicit targets retain their existing
behavior and validation; Ruff's default scope is unchanged.

The literal `.opencode/tools/run_linters.py` now reports
`RESULT: ALL LINTERS PASSED`. All 46 lint-runner tests pass, including real-mypy
regressions that preserve detection of errors in the configured package and
explicit targets despite duplicate unrelated source/payload files.
All required checks now pass at the lint-fix revision, as recorded below.
Historical failures remain below for provenance. Kyle approved the final M1
handoff on 2026-11-10; all M1 gates are satisfied.

## Handoff decision — approved 2026-11-10

**M1 complete (Shipped); M2 unblocked (Ready, not started).** Kyle Gorkowski
responded **“I approve”** in this chat to the explicit question asking for final
M1 handoff approval after all required checks passed. This is final handoff
sign-off, separate from the September P1/P2 scientific/specification approvals.

Approved PR #1624 revision: `a7c1eb63c67f7dc3ac2e35859c9c08de5f44cfd0`.
Implementation validation: `0db7e42b27b0d031c2680af037ea11e7776641b0`;
the approved follow-up changed only evidence Markdown and passed strict docs
checks. All required results, including the default lint runner, are below.
This subsequent approval/status recording changes no implementation or tests.

M2 may begin from the approved accumulation branch with the recorded contract,
helpers and replacement specification. PR merge remains a separate action;
this approval does not claim the PR has merged or a release has shipped.
M2–M6 implementation is not completed by this handoff.

## Preliminary restricted-environment checks

These were run against the P5 working tree based on `c051f27b1`, before the
final committed host-access run. They are diagnostic evidence only: CUDA was
hidden, and the integration accounting assertion was strengthened during this
pass and subsequently passed in the separate 154-test host-access focused run.

- Untargeted runner: 5,857 passed, 135 failed, 163 skipped; 91% coverage,
  threshold 80%. Failed assertion evidence is not a passing full-suite gate.
- Package mypy: `Success: no issues found in 502 source files`.
- Documentation validation-only and strict MkDocs: exit 0 for both.
- Source examples: 148 passed, 1 failed, 2 skipped. The failure compared expected
  no-CUDA warning text with the restricted device-access CUDA error text.
- A failed condensation case rerun with host device/cache access passed (1 test).
  This does not establish the cause or resolution of every preliminary failure.

The JSON maintenance schema check also reports the pre-existing extra
`lifecycle` property on both baseline and edited metadata; the set of schema
errors is unchanged. This is supplemental evidence, not a required P5 gate.

## Committed implementation evidence — 28acda48b

Tested revision: `28acda48b417ef68a6d8d4b9d0c553716e02ea8b`, 2026-11-10.
All commands ran from the repository with `.venv/bin` first on PATH and host
GPU/cache access. CPU and CUDA were available. This evidence-only update does
not change tested production code or assertions. Final PR-head verification
and CI results are published in [PR #1624](https://github.com/uncscode/particula/pull/1624).

| Literal command | Exit | Literal result excerpt |
|---|---|---|
| `pytest particula/integration_tests/data_container_contract_test.py -q --no-cov` | 0 | 4 passed in 0.01s |
| `pytest particula/particles/tests/ particula/gas/tests/ -q --no-cov` | 0 | 682 passed in 0.29s |
| `pytest particula/tests/aerosol_validation_test.py -q --no-cov` | 0 | 51 passed in 0.04s |
| `pytest particula/gpu/tests/conversion_test.py particula/gpu/tests/kernel_exports_test.py -q --no-cov` | 0 | 150 passed in 4.70s |
| `pytest particula/dynamics/tests/wall_loss_strategies_test.py -q --no-cov` | 0 | 16 passed in 0.17s |
| `pytest particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py -q --no-cov` | 0 | 43 passed in 19.01s |
| `.opencode/tools/run_pytest.py` | 0 | Tests Run: 6771; Passed: 6771; Skipped: 19; Coverage: 92% (threshold: 80.0% PASSED); VALIDATION: PASSED |
| `.opencode/tools/run_linters.py` | 1 | Ruff Check: No issues found; Mypy: Duplicate module named "doctor"; RESULT: LINTING FAILED |
| `mypy particula/ --ignore-missing-imports` | 0 | Success: no issues found in 502 source files |
| `python3 .opencode/tools/build_mkdocs.py --validate-only --strict` | 0 | Outcome: success; Stage: complete; Exit Code: 0 |
| `mkdocs build --strict` | 0 | INFO    -  Documentation built in 1.38 seconds |
| `pytest examples_tests -q -Werror` | 0 | 150 passed, 1 skipped in 30.12s |
| `pytest scripts/tests -q -Werror` | 0 | 127 passed in 25.67s |

The successful untargeted run meets the configured coverage policy. Skips do
not represent measured GPU success; the available CUDA conversion cases were
collected and passed (150 conversion/export tests versus 134 with restricted
CPU-only visibility). No benchmark or GPU PDF execution is claimed.

### Historical required failure — resolved by 0db7e42b2

The literal default lint runner calls mypy on the repository root and reports:

```text
plugins/particula-helper/scripts/doctor.py: error: Duplicate module named "doctor" (also at "./plugins/particula-helper/opencode/payload/scripts/doctor.py")
```

Both files match upstream main byte-for-byte (SHA-256
`86b09da1afc3d1cf9ca2241d2b61df6f1e5069e0c83faa0d8b281ca4d7eeaf50`),
at the originally tested revision. The original root-target command overrode
the configured package scope. The later runner fix resolves this discovery
conflict by honoring that existing configuration; neither plugin packaging nor
`pyproject.toml` changes. This historical failure does not override the updated
literal default-runner result recorded above.

## Lint-fix revision — all required checks passed

Revision: `0db7e42b27b0d031c2680af037ea11e7776641b0`, 2026-11-10.
Repository `.venv` with host CUDA/cache access; device details are unchanged.
The following evidence supersedes the earlier failed lint gate. The subsequent
ledger-only commit changes no implementation or tests; its documentation
checks and exact revision are recorded in PR #1624.

| Literal command | Exit | Result |
|---|---|---|
| `pytest particula/integration_tests/data_container_contract_test.py -q --no-cov` | 0 | 4 passed in 0.01s |
| `pytest particula/particles/tests/ particula/gas/tests/ -q --no-cov` | 0 | 682 passed in 0.25s |
| `pytest particula/tests/aerosol_validation_test.py -q --no-cov` | 0 | 51 passed in 0.04s |
| `pytest particula/gpu/tests/conversion_test.py particula/gpu/tests/kernel_exports_test.py -q --no-cov` | 0 | 150 passed in 4.73s |
| `pytest particula/dynamics/tests/wall_loss_strategies_test.py -q --no-cov` | 0 | 16 passed in 0.18s |
| `pytest particula/dynamics/wall_loss/tests/wall_loss_strategies_test.py -q --no-cov` | 0 | 43 passed in 9.01s |
| `.opencode/tools/run_pytest.py` | 0 | 6,771 passed; 19 skipped; 92% coverage (80% threshold); VALIDATION: PASSED |
| `.opencode/tools/run_linters.py` | 0 | Ruff Check: No issues found; Mypy: No issues found; RESULT: ALL LINTERS PASSED |
| `mypy particula/ --ignore-missing-imports` | 0 | Success: no issues found in 502 source files |
| `python3 .opencode/tools/build_mkdocs.py --validate-only --strict` | 0 | Outcome: success; Stage: complete; Exit Code: 0 |
| `mkdocs build --strict` | 0 | INFO    -  Documentation built in 1.37 seconds |
| `pytest examples_tests -q -Werror` | 0 | 150 passed, 1 skipped in 29.94s |
| `pytest scripts/tests -q -Werror` | 0 | 127 passed in 24.82s |
| `pytest .opencode/tools/tool_tests/run_linters_test.py -q --no-cov` | 0 | 46 passed in 1.60s |
