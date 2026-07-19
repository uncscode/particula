# Overview

## Problem Statement

E5's executable coagulation tracks add charged, sedimentation, turbulent-shear,
and approved additive mechanisms, but their phase-local tests do not by
themselves provide one auditable release matrix. Without cross-mechanism
evidence, formula regressions, stochastic bias, conservation failures, or
device-specific gaps can be hidden by uneven fixtures and assertions.

## Value Proposition

E5-F7 publishes a shared, independent validation matrix for every shipped
single-mechanism and approved combined row. It separates deterministic physics
parity, bounded stochastic behavior, mass and charge conservation, ownership
invariants, edge cases, and device coverage so each claim has an explicit
pass/fail result. Warp CPU is the required baseline when Warp is installed;
CUDA is optional additive evidence and skips cleanly when unavailable.

## Implementation Status

Issue #1362 completed the deterministic P1 slice. It adds private test-only
fixture/oracle support and a collection-safe Warp-CPU observation matrix for
the literal executable masks `1`, `2`, `3`, `4`, `5`, `6`, `8`, `9`, `10`,
`12`, and `15`; deferred three-way masks `7`, `11`, `13`, and `14` are covered
as host-only configuration failures. No production behavior, public API, or
 user documentation changed; later phases supplied conservation, ownership,
 stochastic, device, and published-evidence coverage.

Issue #1363 completed the test-only P2 public-entry invariant slice in
`particula/gpu/kernels/tests/coagulation_validation_test.py`. It exercises all
11 executable masks with one- and two-box, one- and two-species
materializations; verifies per-box/per-species inventory, applicable charge,
legal accepted-pair, inactive-slot, and merge bookkeeping invariants; and
checks caller-owned collision/count sidecars and persistent RNG lifecycle.
It also covers turbulent scalar and device-array forms plus selected deferred
and invalid-input preflight non-mutation cases. The suite runs on Warp CPU and
adds CUDA coverage only when the shared device helper reports it. No production
 behavior or public API changed; later phases supplied bounded stochastic and
 published-evidence coverage.

Issue #1364 completed P3 with a bounded fresh-seed public-step stochastic
matrix in `particula/gpu/kernels/tests/coagulation_stochastic_validation_test.py`.
It covers all 11 executable masks on required Warp CPU and optional CUDA,
using 100 unique fresh seeds per row/device, a one-proposal capacity, and an
independent initial-state aggregate expectation with a
`3 * sqrt(expected_mean)` bound. P2 public-step materialization, snapshots,
and invariants were extracted to the private lazy-Warp
`_coagulation_public_step_support.py`; P3 adds explicit volume/concentration
 override regressions while preserving inactive sentinels. No production API,
 kernel, export, shared CUDA helper, or user documentation changed.

Issue #1365 completed P4 and shipped the development documentation, roadmap
evidence, focused commands, and plan-state reconciliation. E5-F7 is complete:
issues #1362--#1365 fulfilled P1--P4, and no E5-F7 work remains.

## User Stories

- As a scientific user, I want independently calculated pair-rate parity so I
  can trust that each GPU mechanism implements its documented equation.
- As a maintainer, I want one parameterized matrix for multi-box, inactive-slot,
  conservation, buffer, and RNG behavior so regressions fail consistently.
- As a release reviewer, I want bounded stochastic and device evidence that
  states exactly what was tested without claiming exact CPU/Warp pair replay.

Parent epic: **E5**. Track: **T7**. Classifier diagnostics: **none**.
