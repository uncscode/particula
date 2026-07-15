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
- [x] Construct the replay harness with device-to-device reset and comparison
  helpers, but do not claim replay evidence: Warp CPU graph capture is skipped
  and the public CUDA step has an expected capture failure from host validation
  readback.
- [x] Assert stable identity, canonical shape, `wp.float64` dtype, and active
  device placement for every supplied scratch sidecar and energy output in
  normal calls; replay assertions remain unreachable until capture support is
  implemented.
- [x] Exercise unsupported capture capability evidence on Warp CPU and optional
  CUDA, with explicit skips/expected failures and no replay-success claim.
- [x] Add `particula/gpu/kernels/tests/condensation_autodiff_test.py` with a
  deterministic one-box/one-particle/one-species, out-of-place smooth-interior
  raw-rate Tape probe. It compares the gas-concentration derivative to a
  centered fp64 finite difference on Warp CPU and optionally CUDA.
- [x] Enable `verify_autograd_array_access` only within the isolated probe and
  restore its exact prior value, including after a sentinel exception. Cover P2
  evaporation clamp, uptake inventory scaling, and in-place mutation as
  forward-only limitation semantics rather than gradient claims.

## Documentation / Validation

- [ ] Publish an evidence matrix, limitations, and focused CPU/CUDA commands.
- [ ] Run focused condensation tests and documentation link validation.
