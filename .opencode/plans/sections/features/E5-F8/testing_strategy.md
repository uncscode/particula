# Testing Strategy

Every phase ships its tests in the same change. Coverage thresholds remain
unchanged, changed executable code must maintain at least 80% coverage, and all
test modules use the `*_test.py` suffix.

## Per-Phase Coverage

- **P1 — Independent walkthrough:** Add
  `particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py`. Assert that
  CPU and Warp builders allocate distinct state, expected values are computed
  before Warp mutation, the no-Warp outcome is explicit, Warp CPU runs whenever
  installed, and unavailable CUDA skips cleanly.
- **P2 — Separate criteria:** In the same test module, verify final particle
  mass, gas concentration, and total transfer physics parity at
  `rtol=2e-10, atol=1e-30`, with constant vapor pressure checked exactly. Verify
  concentration-weighted
  per-box/per-species inventory conservation at `rtol=1e-12, atol=1e-30`, and
  signed energy identity at `rtol=1e-12, atol=1e-18`. Perturb each category in
  isolation so a physics pass cannot satisfy conservation or energy, and vice
  versa. Preserve canonical per-field physics tolerances from the shipped
  parity tests.
- **P3 — Ownership record:** Add
  `particula/tests/condensation_parity_walkthrough_docs_test.py`. Parse the
  record and assert every required deferred capability has exactly one non-empty
  future plan family, entry gate, and explicit non-claim. Thermal feedback and
  adaptive stepping require a future approved condensation numerical-method
  plan; phase-aware surface tension and BAT require a future approved
  condensation-physics plan. Assign no speculative plan IDs and do not broaden
  Epic F.
- **P4 — Documentation integration:** Extend documentation checks for all
  canonical inbound/outbound links, exact focused commands, Warp CPU/CUDA
  policy, two-item return wording, and separate evidence labels.

## Evidence Policy

1. Physics compares Warp observations to an independently evaluated NumPy
   fixed-four-substep oracle; it is not high-level CPU-strategy parity.
2. Conservation recomputes initial and final inventory from independently
   retained state and does not infer conservation from equal oracle outputs.
3. Energy recomputes signed finalized transfer times latent heat and does not
   infer energy correctness from mass conservation.
4. All three required categories must pass. Diagnostics for all categories are
   emitted even when one fails.
5. Warp CPU is the normal required backend when Warp is installed. Optional
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
