# Architecture and Design

## Current Design Boundary

The current low-level coagulation kernel maps one Warp thread to one simulation
box. Inside that thread, the kernel computes active-particle bounds, estimates
trial counts from `k_max`, uses one RNG state per box, and processes stochastic
pair selection sequentially. This deliberately avoids cross-thread races around
collision-pair writes and keeps the API compatible with independent-box GPU
workloads.

## Shipped Decision

E3-F3 does not change the production architecture. The shipped outcome is the
accepted-with-caveat option documented from the existing measurements:

1. **Accepted with caveat:** one-thread-per-box remains the documented low-level
   coagulation baseline for many-box workloads, Warp-backed direct-kernel use,
   and CUDA-backed benchmark/study workflows.
2. **Large single-box caveat stays explicit:** roadmap and foundations docs now
   point back to the measured decision record instead of implying a universal
   production recommendation.
3. **No implementation follow-up was started here:** any future parallel-within-
   box work remains a separate track rather than part of this docs-only issue.

## Boundary Conditions

- CUDA remains optional. Benchmarks and tests must skip cleanly without CUDA.
- No hidden CPU/GPU transfer or synchronization behavior is introduced.
- No graph-capture production optimization is implemented in this feature.
- Documentation must distinguish low-level experimental GPU kernels from stable
  high-level production APIs.
