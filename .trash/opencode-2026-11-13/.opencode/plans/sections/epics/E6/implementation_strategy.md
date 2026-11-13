# Implementation Strategy

## Architecture Overview

Implement process contracts in dependency order: establish NumPy/strategy-
based CPU references, then port bounded physics to direct fixed-shape Warp
kernels. Dilution updates particle and gas concentrations without changing
particle mass, charge, density, or volume. Wall loss uses existing CPU
coefficient models as deterministic oracles and persistent RNG for stochastic
removal. Nucleation converts inventory-limited gas demand into particle source
mass through shared activation and exhaustion services.

GPU entry points remain low-level and explicit. They accept CPU scalar or
device-resident per-box state only where documented, validate all inputs before
allocation or mutation, and never transfer or fall back implicitly.

## Data Ownership

- `ParticleData`, `GasData`, and CPU environment state remain CPU-owned;
  `WarpParticleData`, `WarpGasData`, and Warp environment mirrors remain
  caller-owned device state.
- Particle arrays retain stable shapes. Activation fills inactive slots;
  deactivation clears all required fields; no GPU append or resize is allowed.
- RNG state, diagnostics, scratch storage, and policy controls are explicit,
  caller-owned sidecars and retain identity when supplied.
- Gas depletion and particle source mass finalize per box/species before
  mutation, preventing negative inventory and partial invalid updates.

## Reusable Patterns

- Strategy-backed runnable and substep behavior from
  `particula/dynamics/particle_process.py`.
- Explicit conversion boundaries from `particula/gpu/conversion.py`.
- Fixed-slot clearing and persistent RNG from direct coagulation kernels.
- Gas-coupled inventory finalization and scratch-sidecar validation from direct
  condensation.
- CPU wall-loss strategies as the coefficient and fallback oracle.

## Testing Requirements

1. Test coverage thresholds must NEVER be lowered
2. Each phase must include self-contained tests
3. Tests are committed in the same PR as the implementation
4. Test files use `*_test.py` suffix in module-level `tests/` directories
5. Minimum 80% coverage (configured in `pyproject.toml`)

Each feature includes co-located fast tests. Required evidence includes no-op
and preflight immutability cases, scalar/per-box and multi-box parity,
inactive/full-slot cases, per-box/species conservation, deterministic
coefficient comparisons, statistical stochastic bounds, required Warp CPU
runs, and optional cleanly-skipping CUDA runs.
