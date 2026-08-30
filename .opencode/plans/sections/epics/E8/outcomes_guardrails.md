# Outcomes and Guardrails

- **Primary Outcome:** A supported CUDA path replays fixed-order,
  fixed-shape, GPU-resident multi-box timesteps from a captured Warp graph and
  matches the CPU and uncaptured GPU reference contracts.
- **Secondary Goals:**
  - Separate one-time setup, validation, allocation, and capture from replay.
  - Reuse all required process, communication, diagnostic, and RNG buffers.
  - Publish opt-in scaling and launch-overhead benchmarks across box count and
    particles per box.
  - Publish a reproducible memory-budget model and CUDA profiling evidence.
- **Guardrails / Non-Goals:**
  - No hidden CPU/GPU transfer, synchronization, fallback, device selection,
    retry, or automatic recapture.
  - No dynamic array shapes, process-order changes, or communication-map
    changes during replay; these are explicit recapture triggers.
  - No resizing or compaction; active-count changes use fixed-capacity slots
    and the shipped exhaustion policies.
  - No claim that Warp CPU supports graph capture; it remains the uncaptured
    parity baseline, while capture tests are CUDA-gated.
  - No distributed execution, broad autodiff integration, new physics, or
    redesign of the direct-kernel and resident public boundaries.
  - No performance guarantee without recorded hardware, software, command,
    fixture, and artifact metadata.
