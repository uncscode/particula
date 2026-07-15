# Implementation Tasks

## GPU Test Infrastructure

- [x] Extend `_condensation_test_support.py` with the independent NumPy
  four-substep/P2/gas-coupled expected-output builder used by P1.
- [x] Build two explicit fp64 fixtures covering one/multiple boxes and species,
  uptake, evaporation, partitioning gates, latent heat, gas-limited uptake,
  zero gas, and inactive particle slots.
- [x] Reuse `warp_devices(wp)` at execution time so Warp CPU runs and CUDA is
  optional.
- [x] Export the support cases through `condensation_test.py` as discoverable
  Warp-CPU and CUDA parity tests.

## Correctness Evidence

- [x] Add explicit CPU/Warp particle-mass and gas-concentration parity
  assertions with `rtol=1e-10` and a separate scale-derived finite `atol`.
- [x] Add separate per-box/per-species concentration-weighted particle-plus-gas
  conservation checks.
- [x] Verify transfer, gas loss, particle gain, and E4-F4 latent energy all use
  the same finalized bounded transfer; energy remains unweighted by particle
  concentration.
- [x] Verify inactive/disabled/zero-concentration entries and immutable inputs
  remain unchanged.
- [x] Verify representative wrong shape/dtype/device/configuration paths fail
  before mutation, and verify deterministic fresh runs and caller-owned output
  identities.

## Graph and Autodiff Readiness

- [x] In the discoverable
  `particula/gpu/kernels/tests/condensation_graph_capture_test.py`, construct
  complete E4-F3 caller-owned scratch and fp64 inputs before capture; the
  captured callable contains only the already-created public step call.
- [x] Capture and replay the public `condensation_step_gpu()` four-substep path
  twice with device-to-device resets, then compare each replay to an independent
  normal launch for particle, gas, transfer, and energy with separate parity
  and conservation assertions.
- [x] Assert stable identity, canonical shape, `wp.float64` dtype, and active
  device placement for every supplied scratch sidecar and energy output before
  and after capture and replays.
- [x] Exercise capture readiness on Warp CPU and optional CUDA; unsupported
  capture APIs/capabilities skip with device and operation context, while
  normal-launch and post-launch correctness failures propagate.
- [ ] Add a deterministic, out-of-place smooth-interior tape/gradcheck probe in
  a focused `*_test.py` module only where Warp supports it; cap the test helper
  at a small one-box/one-species fixture rather than expanding production code.
- [ ] Enable array-access verification in that isolated probe and assert explicit
  expected limitations for clamps, inventory gates, and in-place mutation rather
  than weakening the supported-interior assertions.

## Documentation / Validation

- [ ] Publish an evidence matrix, limitations, and focused CPU/CUDA commands.
- [ ] Run focused condensation tests and documentation link validation.
