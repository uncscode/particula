# Testing Strategy

Every implementation phase ships with its tests; coverage thresholds remain
unchanged and changed code must retain at least 80% coverage.

## Per-Phase Approach

- **P1:** In `_condensation_test_support.py` plus `condensation_test.py`, compare
  deterministic fp64 production outputs to independent one-box CPU equations.
  Run one/multi-box and multi-species cases on Warp CPU and optional CUDA with
  explicit physics `rtol`/`atol`.
- **P2:** Add separate strict per-box/per-species inventory and latent-energy
  assertions, targeting `rtol=1e-12` where scale analysis supports the existing
  precedent. Cover inactive entries, clamp bounds, deterministic replay,
  immutable inputs, and fail-before-mutation paths.
- **P3:** Add a discoverable graph-readiness `*_test.py` module. Preallocate all
  scratch, capture exactly four substeps, replay repeatedly, and compare with a
  normal launch for state, transfer, conservation, identity, and shape.
- **P4:** Add an isolated autodiff-readiness `*_test.py` module for a bounded
  smooth-interior tape/gradcheck and access verification. Unsupported backends
  skip with precise reasons; clamp points are tested/documented as non-smooth,
  not assigned permissive gradient tolerances.
- **P5:** Validate documentation links and execute focused reproduction commands.

Parity and conservation are always distinct assertions. Warp absence and CUDA
absence skip cleanly according to repository policy, but when Warp is installed
the CPU backend is mandatory. No benchmark or stochastic marker is required.
