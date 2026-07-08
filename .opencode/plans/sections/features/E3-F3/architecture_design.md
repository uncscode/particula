# Architecture and Design

## Current Design Under Review

The current low-level coagulation kernel maps one Warp thread to one simulation
box. Inside that thread, the kernel computes active-particle bounds, estimates
trial counts from `k_max`, uses one RNG state per box, and processes stochastic
pair selection sequentially. This deliberately avoids cross-thread races around
collision-pair writes and keeps the API compatible with independent-box GPU
workloads.

## Decision Shape

E3-F3 does not change the production architecture. It records an architecture
decision based on refreshed measurements:

1. **Accepted for Epic C:** one-thread-per-box remains the supported low-level
   coagulation design for many-box and experimental CUDA workflows, with
   documented single-box limits.
2. **Accepted with caveat:** the current path remains available, but docs label
   large single-box workloads as a known limitation and point to future work.
3. **Follow-up required:** a separate parallel-within-box feature is scoped for
   implementation after Epic C blockers, without merging optimization work into
   this feature.

## Boundary Conditions

- CUDA remains optional. Benchmarks and tests must skip cleanly without CUDA.
- No hidden CPU/GPU transfer or synchronization behavior is introduced.
- No graph-capture production optimization is implemented in this feature.
- Documentation must distinguish low-level experimental GPU kernels from stable
  high-level production APIs.
