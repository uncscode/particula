# Implementation Tasks

## GPU Test Infrastructure

- [ ] Extend `_condensation_test_support.py` with independent per-box CPU
  reference assembly for all E4-F1 through E4-F5 physics.
- [ ] Build explicit fp64 fixtures covering one/multiple boxes and species,
  condensation/evaporation, partitioning gates, thermal coupling, and clamps.
- [ ] Reuse `warp_devices(wp)` so CPU always runs and CUDA is optional.
- [ ] Keep support cases discoverable through focused `*_test.py` wrappers.

## Correctness Evidence

- [ ] Add explicit CPU/Warp parity assertions with reviewed `rtol` and `atol`.
- [ ] Add separate per-box/per-species particle-plus-gas conservation checks.
- [ ] Verify transfer, gas loss, particle gain, and E4-F4 latent energy all use
  the same finalized bounded transfer.
- [ ] Verify inactive/disabled entries and immutable inputs remain unchanged.
- [ ] Verify wrong shape/device/configuration fails before allocation or mutation.

## Graph and Autodiff Readiness

- [ ] In `particula/gpu/kernels/tests/condensation_test.py` or a new discoverable
  `condensation_graph_test.py`, construct complete E4-F3 caller-owned scratch
  and fixed fp64 inputs before entering capture; do not allocate or transfer
  inside the captured region.
- [ ] Capture and replay the public `condensation_step_gpu()` four-substep path,
  then compare normal-launch and replay particle, gas, transfer, and energy
  outputs with separate parity and conservation assertions.
- [ ] Assert stable identity, shape, dtype, and active-device placement for every
  supplied scratch sidecar across repeated normal and replay calls.
- [ ] Add a deterministic, out-of-place smooth-interior tape/gradcheck probe in
  a focused `*_test.py` module only where Warp supports it; cap the test helper
  at a small one-box/one-species fixture rather than expanding production code.
- [ ] Enable array-access verification in that isolated probe and assert explicit
  expected limitations for clamps, inventory gates, and in-place mutation rather
  than weakening the supported-interior assertions.

## Documentation / Validation

- [ ] Publish an evidence matrix, limitations, and focused CPU/CUDA commands.
- [ ] Run focused condensation tests and documentation link validation.
