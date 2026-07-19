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
- **P3 — Ownership record (not implemented):** Add
  `particula/tests/condensation_parity_walkthrough_docs_test.py`. Parse the
  record and assert every required deferred capability has exactly one non-empty
  future plan family, entry gate, and explicit non-claim. Thermal feedback and
  adaptive stepping require a future approved condensation numerical-method
  plan; phase-aware surface tension and BAT require a future approved
  condensation-physics plan. Assign no speculative plan IDs and do not broaden
  Epic F.
- **P4 — Documentation integration (not implemented):** Extend documentation checks for all
  canonical inbound/outbound links, exact focused commands, Warp CPU/CUDA
  policy, two-item return wording, and separate evidence labels.

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
pytest particula/gpu/kernels/tests/condensation_test.py -q -Werror
python docs/Examples/gpu_condensation_parity_walkthrough.py
```

The full repository test and documentation validation remain final regression
gates; no existing threshold or marker policy may be weakened.
