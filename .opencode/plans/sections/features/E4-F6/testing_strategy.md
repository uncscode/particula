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
- **P2 (completed, #1309):** `_condensation_test_support.py` adds Warp-CPU
  public-step contract regressions, re-exported by `condensation_test.py`.
  They assert per-box/per-species concentration-weighted particle-plus-gas
  conservation, returned total-transfer accounting, and unweighted latent
  energy from finalized transfer at `rtol=1e-12` and
  `atol=max(1e-18, scale * eps)`. Coverage includes inactive, disabled, and
  zero-concentration entries; inventory-limited uptake; finite/nonnegative
  state; immutable inputs; deterministic fresh runs; caller-owned output
  identity; and atomic representative invalid-buffer/configuration paths.
- **P3 (completed, #1310):**
  `particula/gpu/kernels/tests/condensation_graph_capture_test.py` preallocates
  all seven scratch fields, latent heat, energy, and device-resident mutable
  reset sources. It captures only the already-created public four-substep call,
  restores mutable inputs/outputs device-to-device, and compares two launches
  of the same graph with an independent normal launch for mass, gas, transfer,
  and energy at `rtol=2e-10`, `atol=1e-30`. Normal and replay states separately
  pass strict per-box/per-species inventory and energy assertions; sidecar
  identity/shape/dtype/device stability is checked throughout. Warp CPU is
  covered and CUDA is optional. Capture support failures skip with the device
  and failed capture operation; correctness failures do not skip.
- **P4 (completed, #1311):**
  `particula/gpu/kernels/tests/condensation_autodiff_test.py` differentiates a
  one-box, out-of-place raw `condensation_mass_transfer_kernel` proposal with
  respect to gas concentration. It compares Warp Tape with a centered fp64
  reference at `rtol=2e-5`, `atol=1e-18`, with executable positive/inventory
  margins. The probe scopes `verify_autograd_array_access` and verifies exact
  restoration after an exception. Warp CPU is required when available and CUDA
  adds the same optional evidence; absent Tape, backward, or access-verification
  capability skips with precise bounded-probe/device context. Separate
  forward-only tests cover P2 evaporation clamp, inventory scaling, and
  in-place mutation without asserting backend-specific gradients or warnings.
- **P5:** Validate documentation links and execute focused reproduction commands.

Parity and conservation are always distinct assertions. Warp absence and CUDA
absence skip cleanly according to repository policy, but when Warp is installed
the CPU backend is mandatory. No benchmark or stochastic marker is required.
