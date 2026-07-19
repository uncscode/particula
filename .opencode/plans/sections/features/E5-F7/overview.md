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
user documentation changed. Conservation, ownership, stochastic, CUDA, and
published-evidence work remain subsequent phases.

## User Stories

- As a scientific user, I want independently calculated pair-rate parity so I
  can trust that each GPU mechanism implements its documented equation.
- As a maintainer, I want one parameterized matrix for multi-box, inactive-slot,
  conservation, buffer, and RNG behavior so regressions fail consistently.
- As a release reviewer, I want bounded stochastic and device evidence that
  states exactly what was tested without claiming exact CPU/Warp pair replay.

Parent epic: **E5**. Track: **T7**. Classifier diagnostics: **none**.
