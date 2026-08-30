# Vision and Problem

Particula now has a shipped GPU-resident, multi-box execution system, but its
repeated timestep path still pays uncaptured launch and host-orchestration
costs. Performance and memory behavior are not yet characterized as a coherent
end-to-end contract. This creates four program-level problems:

1. **Launch overhead remains visible** -- small and medium repeated workloads
   dispatch the fixed process sequence one operation at a time.
2. **Capture boundaries are undefined** -- allocation, validation, scheduling,
   diagnostics, RNG setup, and mutable state have not been divided into setup
   work and graph-replay-safe work.
3. **Capacity planning lacks evidence** -- users do not have published scaling
   results or a memory model covering boxes, particles, species, inactive
   slots, communication, diagnostics, and future autodiff tape storage.
4. **Optimization claims are not integrated** -- existing kernel benchmarks do
   not prove captured full-loop correctness or end-to-end benefit.

## The Vision

A caller performs explicit setup once, captures a fixed-shape and fixed-order
GPU-resident timestep on supported CUDA devices, and replays it without hidden
transfer, allocation, synchronization, or RNG reseeding. Captured results match
the CPU and uncaptured GPU references within documented scientific tolerances.
Published multi-box scaling, launch-overhead, profiling, and memory-budget
evidence lets users decide when the path is appropriate.

## Why Now

Epic G established the stable resident session, scheduler, sidecar ownership,
communication maps, checkpoint/restart policy, and complete-loop validation
that graph capture requires. Epic H is therefore the next bounded roadmap step
before broader differentiability and optimization work can rely on predictable
execution and memory behavior.
