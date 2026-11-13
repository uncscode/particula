# Implementation Strategy

## Architecture Overview

Retain `particula/gpu/kernels/coagulation.py` as the low-level orchestration
boundary and factor mechanism-specific pair calculations into focused GPU
helpers alongside the existing Brownian implementation. Mechanism
configuration selects approved pair-rate terms; combined execution adds those
terms for a candidate pair, computes a proven total majorant, and performs one
acceptance/sampling pass. Existing CPU implementations provide independent
formula references rather than shared expected-value code.

The implementation sequence is contract first, individual physics second,
additive composition third, and release evidence/documentation last. Charged,
sedimentation, and turbulent-shear functions ship with co-located unit tests.
Cross-mechanism integration evidence complements those tests; it does not defer
unit validation to a standalone testing phase.

## Data Ownership Rules

- Callers continue to own `WarpParticleData`, fixed-shape work buffers, and
  optional persistent `rng_states`; the step mutates only documented fields.
- Particle charge remains caller-owned in `WarpParticleData.charge` as
  dimensionless elementary-charge counts. It is fp64, fixed-shape,
  active-device, and finite before downstream work; a merge adds donor charge
  to the recipient and clears the donor alongside its mass/state.
- Mechanism-specific scalar or per-box inputs are explicitly supplied,
  validated on the active device, and never fetched through hidden transfers.
- Combined mechanisms share one candidate/acceptance pass; they do not advance
  separate stochastic streams per mechanism.
- Unsupported distributions, variants, devices, shapes, and values fail before
  particle mutation or RNG advancement.

## Reusable Codebase Patterns

- Validation, allocation, launch, persistent-RNG, and buffer patterns from
  `particula/gpu/kernels/coagulation.py`.
- Focused Warp ports in `particula/gpu/dynamics/coagulation_funcs.py`.
- CPU charged merge semantics from
  `coagulation_strategy/coagulation_strategy_abc.py`.
- Device-aware parity and stochastic policy from
  `.opencode/guides/testing_guide.md`.
- Explicit transfer and reusable sidecar flow from
  `docs/Examples/gpu_direct_kernels_quick_start.py`.

## Testing Requirements

1. Test coverage thresholds must NEVER be lowered
2. Each phase must include self-contained tests
3. Tests are committed in the same PR as the implementation
4. Test files use `*_test.py` suffix in module-level `tests/` directories
5. Minimum 80% coverage (configured in `pyproject.toml`)

Additionally, pair formulas use deterministic CPU/Warp fixtures; stochastic
execution uses aggregate or sigma-based bounds rather than exact pair replay;
mass and charge conservation are asserted separately; inactive slots,
multi-box inputs, persistent RNG, and caller-owned buffers are covered. Warp
CPU is required when Warp is installed, while CUDA remains optional and must
skip cleanly when unavailable.
