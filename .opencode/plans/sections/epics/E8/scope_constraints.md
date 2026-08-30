# Scope and Constraints

## In Scope

- Warp graph capture and replay for the resident scheduler's fixed repeated
  timestep sequence.
- A strict setup/replay split for allocation, validation, scheduling, resource
  acquisition, stream initialization, graph capture, and replay.
- Preallocated reusable buffers for condensation transfer, coagulation output,
  wall loss, dilution, nucleation/exhaustion, diagnostics, communication, and
  persistent RNG state.
- CPU-versus-uncaptured-versus-captured full-loop correctness coverage.
- Opt-in CUDA scaling benchmarks, graph launch-overhead comparisons, and
  occupancy/memory-access profiling.
- A parameterized memory model for state, inactive capacity, temporary
  storage, communication, diagnostics, checkpoints, and future autodiff tape.
- A complete captured-loop example, limitations, recapture triggers, and
  reproduction commands.

## Out of Scope

- Dynamic process selection or variable shapes inside a captured graph.
- New distributed transport, broad mixing/advection, new aerosol physics, or
  new CPU strategy APIs.
- Transparent capture, automatic recapture, automatic device migration,
  cross-device replay, or CPU fallback.
- End-to-end autodiff or optimization APIs; this epic only records the memory
  allowance needed by later work.

## Constraints

- Preserve explicit caller ownership of resident containers and sidecars.
- Preserve fixed `n_boxes`, `n_particles`, `n_species`, process order,
  communication maps, device, and buffer identities for a capture lifetime.
- Initialize or explicitly reset persistent RNG streams before capture; replay
  must never reseed implicitly.
- Warp CPU remains required for uncaptured parity when Warp is installed.
  Capture and profiling evidence must be CUDA-gated and skip cleanly when no
  qualified CUDA device is available.
- Unit and contract tests ship with every child implementation; benchmarks
  remain opt-in and do not alter default test collection.
