# Vision and Problem

Epic C hardens Particula's low-level GPU kernel layer after the data-container
and foundational GPU work. The current GPU path is useful for direct Warp kernel
experiments, but several correctness and usability gaps make it risky for users
and future epics to build on it without clarifying contracts first.

## Problems Today

1. **Coagulation RNG state resets across repeated calls.**
   `coagulation_step_gpu` accepts caller-owned RNG state but currently
   reinitializes state on every call, which can correlate repeated stochastic
   draws when callers expect seed-once semantics.
2. **Mixed NPF/droplet rejection sampling is under-characterized.**
   The coagulation kernel uses a single global `k_max`; very wide particle-size
   ranges may collapse acceptance rates or need an explicit limitation in docs
   and tests.
3. **The one-thread-per-box design lacks a crisp decision record.**
   The implementation is intentionally simple, but users need measured scaling
   limits and a documented boundary between accepted Epic C behavior and future
   parallel-within-box work.
4. **Low-level kernel import paths are not sufficiently discoverable.**
   Kernels are exported from `particula.gpu.kernels`, while top-level
   `particula.gpu` emphasizes explicit transfer helpers. The intended public
   surface needs documentation or export updates.
5. **Device-aware validation policy is implicit.**
   Warp CPU and CUDA-if-available testing patterns exist, but parity,
   stochastic tolerance, and optional CUDA expectations should be formalized.
6. **Latent-heat examples and baselines remain incomplete.**
   The CPU `CondensationLatentHeat` strategy needs a runnable documentation
   example and a deterministic conservation baseline to support future GPU
   parity work.

## The Vision

After this epic ships, Particula has a hardened low-level GPU kernel contract:

- Repeated coagulation kernel calls can preserve RNG state when callers provide
  it, while default seed behavior remains documented and regression-tested.
- Mixed-scale coagulation sampling behavior is either hardened or explicitly
  bounded by tests, documentation, and follow-up scope.
- Coagulation threading architecture is recorded with benchmark evidence so
  future performance work starts from an agreed baseline.
- Direct GPU kernel usage has a short, runnable quick-start that preserves the
  explicit CPU/GPU transfer boundary.
- Warp CPU and CUDA-if-available tests follow a documented tolerance policy
  that avoids exact stochastic parity where inappropriate.
- Latent-heat CPU examples and integration baselines are available for future
  Epic D GPU parity validation.

## Why Now

Epic C is the bridge between foundational GPU data structures and future higher
level GPU workflows. Hardening correctness, import paths, and validation policy
now prevents later epics from encoding ambiguous behavior as de facto API. The
scope deliberately avoids high-level backend selection, new GPU physics, hidden
CPU/GPU transfers, and required CUDA hardware so the work remains focused and
reviewable.
