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

## Required gates

Final host-access results will be appended after execution. Initial focused CPU checks
passed (integration 4, particle/gas 682, structure 51, conversion/exports 134,
standard wall-loss 16, adjacent wall-loss 43); these working-tree checks are
not final-revision or CUDA evidence. The literal default lint runner reports a
duplicate `doctor` module under assistant-plugin copies. A passing scoped
package mypy check cannot replace that failed required default command.

## Handoff decision

**M1 incomplete; M2 blocked.** P1 scientific and P2 specification approval
was recorded on 2026-09-28. Final M1 approval has not been requested or granted
against the completed P5 diff and final evidence. Required failed/unavailable
checks must be resolved and Kyle must approve the revision-specific handoff
before M1 is marked complete. No M2–M6 work or release completion is claimed.

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
