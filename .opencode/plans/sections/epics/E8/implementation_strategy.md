# Implementation Strategy

## Architecture Overview

Build graph capture as a concrete lifecycle layer around the shipped resident
execution system rather than as a second scheduler. One-time setup validates an
exact active session, device, fixed dimensions, complete resolved schedule,
stable communication configuration, and pinned resource identities. It then
allocates or acquires all sidecars, explicitly initializes selected persistent
RNG streams, and captures the same ordered device operations used by the
uncaptured scheduler. Replay accepts only compatible scalar/control updates and
rejects every structural change before launch.

The implementation should keep three explicit boundaries:

1. **Setup:** host validation, normalization, resource acquisition, allocation,
   stream initialization, and capture.
2. **Replay:** graph launch against the exact bound session, buffers, device,
   shapes, process order, communication map, and RNG resources.
3. **Teardown/recapture:** explicit release or replacement after a documented
   trigger; no automatic migration or hidden fallback.

## Data Ownership Rules

- The active resident session owns primary particle, gas, and environment
  container identities for the capture lifetime.
- `GPUResourceRegistry` remains the authority for stable-shape, same-device,
  nonaliasing process, communication, diagnostic, and RNG sidecars.
- The graph lifecycle record owns compatibility metadata and the captured graph
  handle; it does not own or silently replace caller state.
- Checkpoint and finalize require successful graph teardown first. Checkpoints
  never retain a dormant graph handle, and continuation or restart requires a
  fresh capture against the active session identities.
- Persistent coagulation and wall-loss RNG words advance during replay.
  Initialization/reset is explicit before capture, and checkpoint continuation
  remains governed by the shipped schema-v3 contract.
- Benchmark and profiling artifacts are evidence, not runtime configuration or
  a promise of equivalent performance on other hardware.

## Reusable Codebase Patterns

- Reuse fail-closed identity/device/schema validation from
  `particula.execution.gpu_session`, `gpu_resources`, and `checkpoint`.
- Reuse the exact process ordering and writer-failure semantics of
  `particula.execution.resident_scheduler`.
- Reuse fixed-slot activation/exhaustion instead of changing array capacity.
- Extend existing `benchmark` marker and `--benchmark` gating; do not change
  default pytest collection.
- Record explicit tolerances, conservation checks, deterministic fixtures, and
  aggregate stochastic bounds following current GPU parity policy.

## Testing Requirements

1. Test coverage thresholds must NEVER be lowered
2. Each phase must include self-contained tests
3. Tests are committed in the same PR as the implementation
4. Test files use `*_test.py` suffix in module-level `tests/` directories
5. Minimum 80% coverage (configured in `pyproject.toml`)

Additionally, Warp CPU must cover uncaptured compatibility and rejection
contracts when Warp is installed. CUDA capture rows are pass-or-clean-skip
evidence and may not fall back to CPU. E8-F4 must compare identical process
configurations over multiple timesteps among CPU, uncaptured GPU, and captured
GPU paths, with per-process tolerances, inventory conservation, RNG lifecycle,
fault-state, and recapture-trigger coverage. Benchmarks and profilers remain
separate from the default parity suite.
