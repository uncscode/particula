# Testing Strategy

Every phase ships its tests in the same change. Coverage thresholds remain
unchanged, changed executable code must maintain at least 80% coverage, and all
test modules use the `*_test.py` suffix.

## Per-Phase Coverage

- **P1 — Independent walkthrough (implemented):**
  `particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py` covers
  immutable fp64 templates, detached/non-aliasing builders, four-substep oracle
  diagnostics, no-Warp and force-disabled behavior, fake enabled sidecars and
  synchronization, runtime failures, direct script execution, Warp CPU parity,
  and optional CUDA parity.
- **Implemented tolerances:** Oracle energy identity uses `rtol=1e-12,
  atol=1e-30`; Warp CPU compares masses, gas, total transfer, final raw
  proposal, energy, and concentration-weighted inventory at `rtol=1e-10,
  atol=1e-30`. The CUDA test uses the same parity envelope when available.
- **P2 — Categorized acceptance (implemented):** The same module covers three
  ordered `AcceptanceResult` blocks. Physics uses final mass, gas, and P2 total
  transfer at `rtol=2e-10, atol=1e-30` plus exact vapor pressure; conservation
  recomputes observed concentration-weighted drift at `rtol=1e-12,
  atol=1e-30`; energy recomputes signed P2-transfer times latent heat at
  `rtol=1e-12, atol=1e-18`. No-Warp results are explicitly `unavailable`.
  Isolated vapor-pressure, energy-sidecar, and detached conservation-input
  mutations, plus a multi-failure case, verify all categories are reported.
- **P3 — Ownership record (implemented):**
  `particula/tests/condensation_parity_walkthrough_docs_test.py` parses the sole
  14-row record table and validates each owner, entry gate, and explicit
  non-claim. It also validates exact links, anchors, seven focused commands, and
  the boundary limiting evidence to the independent fp64 fixed-four-substep
  NumPy oracle versus the direct kernel. It rejects speculative plan IDs, Epic F
  ownership, and positive support claims for deferred capabilities.
- **P4 — Documentation integration (implemented):**
  `particula/tests/condensation_parity_walkthrough_docs_test.py` validates the
  two exact once-only resolving links on each of the five canonical pages, the
  walkthrough and focused pytest commands, separate `physics`,
  `conservation`, and `energy` labels, fixed-four-substep low-level direct-kernel
  wording, Warp CPU baseline/optional CUDA policy, and `energy_transfer` as
  caller-owned write-only diagnostic output. Aggregate negative assertions
  reject unsupported positive claims, including high-level strategy/`Runnable`
  parity, temperature feedback, adaptive stepping, graph replay, broad autodiff,
  performance, and required CUDA.

## Evidence Policy

1. Implemented parity compares Warp observations to an independently evaluated NumPy
   fixed-four-substep oracle; it is not high-level CPU-strategy parity.
2. Categorized acceptance independently recomputes concentration-weighted
   inventory and signed energy from Warp observations; neither is inferred from
   oracle parity.
3. Warp CPU is the normal required backend when Warp is installed. Optional
   CUDA is additive and must skip cleanly when unavailable.

## Focused Verification

```bash
pytest particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py -q -Werror
pytest particula/tests/condensation_parity_walkthrough_docs_test.py -q -Werror
pytest particula/gpu/kernels/tests/condensation_test.py -q -Werror
python docs/Examples/gpu_condensation_parity_walkthrough.py
```

The full repository test and documentation validation remain final regression
gates; no existing threshold or marker policy may be weakened.
