# Implementation Strategy

## Architecture

Extend `particula/gpu/kernels/condensation.py` without introducing hidden
transfer boundaries. Numeric model configuration selects supported Warp-safe
thermodynamic formulas. Each of four fixed substeps refreshes vapor pressure
and particle-dependent equilibrium physics, computes bounded transfer, applies
latent correction, updates particle/gas state, and accumulates whole-call
diagnostics. Stable-shape caller-owned scratch avoids repeated allocation and
keeps graph capture feasible.

## Data Ownership

- CPU callers own ordered species names and explicit CPU/Warp conversion.
- `WarpEnvironmentData` owns device temperature, pressure, and saturation ratio.
- `WarpGasData.vapor_pressure` remains GPU helper state refreshed on-device.
- Particle and gas arrays are mutated in place only after full preflight validation.
- Persistent scratch is caller-owned; returned mass/energy diagnostics cover the full call.

## Reused Patterns

- Environment normalization from `particula/gpu/kernels/environment.py`.
- Fixed four-substep prototype in `_condensation_test_support.py`.
- CPU equations in `condensation_strategies.py` and `mass_transfer.py`.
- Explicit transfer and validation patterns in `conversion.py` and current GPU kernels.

## Testing Requirements

1. Test coverage thresholds must NEVER be lowered
2. Each phase must include self-contained tests
3. Tests are committed in the same PR as the implementation
4. Test files use `*_test.py` suffix in module-level `tests/` directories
5. Minimum 80% coverage (configured in `pyproject.toml`)

Each feature PR tests positive and negative paths, shape/device validation, and
its physics invariant. Warp CPU is required when Warp is installed; CUDA skips
cleanly when unavailable. Physics parity uses explicit tolerances while mass
conservation and `Q = delta_mass * latent_heat` remain separately strict.
