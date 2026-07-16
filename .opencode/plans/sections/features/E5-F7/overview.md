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

## User Stories

- As a scientific user, I want independently calculated pair-rate parity so I
  can trust that each GPU mechanism implements its documented equation.
- As a maintainer, I want one parameterized matrix for multi-box, inactive-slot,
  conservation, buffer, and RNG behavior so regressions fail consistently.
- As a release reviewer, I want bounded stochastic and device evidence that
  states exactly what was tested without claiming exact CPU/Warp pair replay.

Parent epic: **E5**. Track: **T7**. Classifier diagnostics: **none**.
