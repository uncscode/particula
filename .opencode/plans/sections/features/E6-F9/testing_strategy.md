# Testing Strategy

Every implementation phase ships with co-located tests. Test files use the
`*_test.py` suffix, the configured coverage threshold remains at least 80%, and
no threshold may be lowered. Scientific expectations come from independent
NumPy/CPU references or direct accounting equations, not the GPU function under
test.

## Per-Phase Approach

- **P1:** `particula/gpu/tests/process_sequence_test.py` validates fixture
  shapes/dtypes, independent per-process budgets, snapshots, and diagnostic
  expectations for one and multiple boxes/species.
- **P2:** The same module executes all direct processes on shared device state.
  Warp CPU is the required installed-Warp baseline; CUDA is optional and skips
  cleanly. Cases cover no-ops, activation, exhaustion-policy outcomes, repeated
  calls, persistent RNG, and invalid preflight immutability.
- **P3:**
  `particula/gpu/tests/gpu_complete_process_sequence_example_test.py` validates
  lazy imports, forced-no-Warp behavior, subprocess output, one initial
  conversion boundary, no intermediate host restore, one final checkpoint,
  direct-call order, and sidecar identity.
- **P4:** Documentation tests validate links, imports, focused commands, E6 and
  E6-F1-F9 inventories, exit-bar wording, and explicit Epic G boundaries. Run
  `adw plans validate` for plan consistency.

## Required Invariants

- Fixed shapes, fp64 physics arrays, documented integer diagnostic dtypes,
  active device, and caller-owned object identities remain stable.
- Condensation and nucleation conserve represented particle plus gas mass per
  box/species at each feature's recorded tolerance; coagulation conserves mass
  and charge; dilution and wall loss match independent expected loss budgets.
- Gas remains finite and nonnegative, inactive/free slot predicates remain
  exact, and no exhausted nucleation demand is silently truncated.
- Deterministic coefficients use recorded `rtol`/`atol`; stochastic removal
  uses predeclared aggregate or sigma bounds, never exact backend RNG equality.
- Invalid calls preserve particles, gas, environment, volume, RNG, requests,
  diagnostics, scratch, and work buffers byte-for-byte where specified.
- Instrumentation rejects any `from_warp_*` call between direct process calls.

## Focused Commands

```bash
pytest particula/gpu/tests/process_sequence_test.py -q -Werror
pytest particula/gpu/tests/gpu_complete_process_sequence_example_test.py -q -Werror
pytest particula/gpu/tests/process_sequence_test.py -q -m "warp and gpu_parity and not cuda"
pytest particula/gpu/tests/process_sequence_test.py -q -m "warp and cuda"
```
