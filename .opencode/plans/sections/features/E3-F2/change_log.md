# E3-F2 Change Log

## 2026-07-08

- Created first-pass feature plan for hardening or characterizing GPU
  coagulation rejection sampling across mixed NPF/droplet size ranges.
- Added four issue-sized phases covering mixed-scale fixtures and diagnostics,
  bounded sampler hardening, statistical/conservation comparison, and final
  documentation.
- Recorded dependency on E3-F1 RNG API compatibility and seed-once
  initialization semantics.

## 2026-07-08 — Completeness Review

- Replaced generic success bullets with measurable pass/fail criteria and
  evidence metrics for fixture coverage, acceptance visibility, conservation,
  stochastic correctness, and final documentation.

## 2026-07-09

- Updated the plan after issue #1241 shipped E3-F2-P1.
- Recorded that the landed implementation stayed entirely in
  `particula/gpu/kernels/tests/coagulation_test.py` with a deterministic mixed
  NPF/droplet fixture, a test-local mirrored attempt diagnostic, and targeted
  mixed-scale Warp CPU/CUDA regression tests.
- Noted that no public `coagulation_step_gpu(...)` API or production
  synchronization behavior changed in this phase.
