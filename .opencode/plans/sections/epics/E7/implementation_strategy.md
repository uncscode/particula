# Implementation Strategy

## Architecture Overview

Introduce a typed execution-context/session layer above existing CPU runnables
and deliberate direct GPU kernel exports. Backend selection resolves a declared
capability; it does not infer fallback from runtime failures. A resident GPU
session owns fixed-shape `WarpParticleData`, `WarpGasData`, and
`WarpEnvironmentData` plus reusable process sidecars, diagnostics, and RNG
streams. Process adapters normalize lifecycle and result semantics without
rewriting kernel physics.

The scheduler builds a deterministic capability graph and executes a canonical
order. Environment changes precede derived thermodynamic refreshes; refreshed
vapor pressure and saturation state precede consuming processes. Multi-box
transport uses a fixed-capacity canonical edge list with canonical
source/destination order and an active-edge count; it is not hidden inside
process kernels. Checkpoint/finalize operations synchronize and restore CPU
state; normal timesteps do neither.

## Data Ownership Rules

- CPU containers remain authoritative before setup and after explicit restore.
- During a GPU session, Warp containers are authoritative mutable state.
- The session owns reusable allocations and lifecycle; configuration remains
  immutable and caller visible.
- RNG state is persistent, per-box, checkpointed state. A repeated seed does
  not silently reset a stream.
- Gas names and other intentionally CPU-owned metadata remain external to Warp
  containers and are carried by checkpoint metadata.
- Adapters return or retain identity according to shipped direct-kernel
  contracts; no intermediate conversion is allowed.

## Reused Repository Patterns

- `particula/runnable.py` for deterministic CPU composition semantics.
- `particula/gpu/warp_types.py` and `particula/gpu/conversion.py` for fixed
  schemas and explicit transfer boundaries.
- `particula/gpu/kernels/__init__.py` for narrow public exports.
- `docs/Examples/gpu_complete_process_sequence.py` as implementation seed,
  not as a production scheduler.
- Existing `process_sequence_test.py`, example regression, kernel export, and
  module-level process tests for parity and no-transfer assertions.

## Testing Requirements

1. Test coverage thresholds must NEVER be lowered
2. Each phase must include self-contained tests
3. Tests are committed in the same PR as the implementation
4. Test files use `*_test.py` suffix in module-level `tests/` directories
5. Minimum 80% coverage (configured in `pyproject.toml`)

Additionally, use independent NumPy/CPU references, explicit tolerances,
particle-plus-gas conservation checks, identity/shape assertions, transfer-call
spies, deterministic call-order tests, restart equivalence tests, isolated-box
metamorphic tests, Warp CPU as baseline, and optional CUDA rows that skip
cleanly. E7-F9 owns cross-feature regressions, but each earlier feature ships
its unit and contract tests with its implementation.
