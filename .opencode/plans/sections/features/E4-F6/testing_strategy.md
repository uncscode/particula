# Testing Strategy

Every implementation phase ships with its tests; coverage thresholds remain
unchanged and changed code must retain at least 80% coverage.

## Per-Phase Approach

- **P1 (completed, #1308):** `_condensation_test_support.py` defines two shared
  deterministic fp64 cases and a NumPy fixed-four-substep/P2/gas-coupled
  expected-output builder; `condensation_test.py` exports the Warp-CPU and CUDA
  matrix tests. The cases independently compare final particle masses and gas
  concentrations at `rtol=1e-10`, with
  `atol=max(max(abs(expected)) * 1e-12, 1e-30)` per output. Warp CPU is required
  when Warp is installed; CUDA uses the same matrix and skips cleanly when
  unavailable.
- **P2:** Add separate strict per-box/per-species inventory and latent-energy
  assertions with `rtol=1e-12` and
  `atol=max(1e-18, scale * eps)`. Cover inactive entries, clamp bounds,
  deterministic replay, immutable inputs, and fail-before-mutation paths.
- **P3:** Add
  `particula/gpu/kernels/tests/condensation_graph_capture_test.py`. Preallocate
  all scratch, capture exactly four substeps, replay repeatedly, and compare
  with a normal launch for state, transfer, conservation, identity, and shape.
- **P4:** Add
  `particula/gpu/kernels/tests/condensation_autodiff_test.py` for a bounded
  out-of-place smooth-interior tape/gradcheck and access verification.
  Unsupported backends skip with precise reasons; clamps, inventory gates, and
  in-place mutation are documented as unsupported rather than assigned
  permissive gradient tolerances.
- **P5:** Validate documentation links and execute focused reproduction commands.

Parity and conservation are always distinct assertions. Warp absence and CUDA
absence skip cleanly according to repository policy, but when Warp is installed
the CPU backend is mandatory. No benchmark or stochastic marker is required.
