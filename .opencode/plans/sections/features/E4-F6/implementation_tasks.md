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

- [ ] Build capture inputs and complete reusable scratch before capture.
- [ ] Capture/replay the fixed four-substep path and compare normal-launch and
  replay outputs plus strict invariants.
- [ ] Assert stable scratch object identity and shape across repeated calls.
- [ ] Add a bounded deterministic tape/gradcheck probe where Warp supports it.
- [ ] Enable array-access verification in an isolated test and document any
  expected in-place or clamp limitation rather than weakening assertions.

## Documentation / Validation

- [ ] Publish an evidence matrix, limitations, and focused CPU/CUDA commands.
- [ ] Run focused condensation tests and documentation link validation.
