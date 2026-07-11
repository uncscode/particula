# E3-F5 Overview: Device-aware GPU pytest policy

## Problem Statement

Epic C is hardening GPU kernel correctness and public low-level API behavior.
Current Warp tests already run on Warp CPU and opportunistically include CUDA
when available, but this behavior is distributed across per-module fixtures and
local skips. The repository lacks a single documented pytest policy for Warp
CPU, CUDA-if-available validation, and stochastic parity tolerances. That makes
it easy for future GPU tests to bypass Warp CPU parity, make CUDA mandatory by
accident, or assert exact equality where stochastic kernels only support
statistical agreement.

## Value Proposition

This feature formalizes project-wide test semantics for GPU kernel validation.
E3-F5-P1 has now shipped the policy foundation: shared pytest marker
registration for `warp`, `cuda`, `gpu_parity`, and `stochastic`; matching
static marker declarations in `pyproject.toml`; and regression tests proving
that `--benchmark` remains the only collection-affecting option. Later phases
still own reusable device helpers, tolerance documentation, and broad marker
adoption across GPU test modules.

## User Stories

- As a contributor adding a Warp kernel test, I want a standard device fixture
  and marker policy so that my test runs on Warp CPU and includes CUDA without
  bespoke skip logic.
- As a maintainer reviewing stochastic coagulation changes, I want documented
  tolerance classes so that tests check aggregate statistical behavior rather
  than exact per-seed equality.
- As a release operator, I want CUDA validation expectations documented as
  optional local/manual checks so that CPU-only CI remains reliable.

## Parent Epic Context

- Parent epic: E3.
- Sibling dependencies: E3-F1 establishes seed-once persisted RNG behavior;
  E3-F2 establishes mixed-scale stochastic sampling evidence. E3-F5 should
  encode those expectations into reusable pytest policy and documentation.

## Shipped Phase Snapshot

- Issue `#1257` / `E3-F5-P1` delivered the hook-level policy foundation only.
- `particula/conftest.py` now centralizes the shared marker vocabulary in
  `PYTEST_MARKER_LINES` and keeps `--benchmark` as the sole pytest option.
- `pyproject.toml` mirrors the exact same marker strings to prevent
  unknown-marker drift.
- Regression coverage now lives in
  `particula/tests/pytest_marker_policy_test.py`, with existing
  `particula/tests/benchmark_option_test.py` kept green for the benchmark
  option path.
