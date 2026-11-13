# Outcomes and Guardrails

- **Primary Outcome:** Provide user-facing CPU/GPU backend selection and a
  deterministic simulation session that keeps particle, gas, environment,
  sidecar, diagnostic, and RNG state resident on the selected device across
  timesteps, with transfers only at explicit checkpoints.
- **Secondary Goals:**
  - Run supported condensation and Brownian coagulation through the selection
    API with CPU-reference parity and documented tolerances.
  - Compose all shipped direct GPU processes in a canonical order that refreshes
    environment and derived gas state before consumers run.
  - Support isolated multi-box execution plus prescribed communication,
    mixing, and volume evolution under explicit conservation rules.
  - Make per-box stochastic streams reproducible across checkpoint/restart and
    insensitive to unrelated box additions, disabling, or reordering.
- **Guardrails / Non-Goals:**
  - No silent fallback, hidden CPU/GPU transfers, or implicit synchronization.
  - No rewrite or unsupported expansion of shipped process physics; no GPU
    staggered condensation or broad BAT parity promise.
  - No dynamic particle resizing or compaction; fixed-slot contracts remain.
  - No multi-GPU/distributed execution or full CFD coupling.
  - No mandatory CUDA CI or exact CPU/CUDA stochastic trajectory equality.
  - No graph capture, profiling, or performance optimization owned by Epic H.
  - No autodiff or optimization work owned by Epic I.
