# Testing Strategy

Every implementation phase ships with co-located tests. Test files use the
`*_test.py` suffix, the configured coverage threshold remains at least 80%, and
no threshold may be lowered. Scientific expectations come from independent
NumPy/CPU references or direct accounting equations, not the GPU function under
test.

## Per-Phase Approach

- **P1 (complete, #1446):** `particula/gpu/tests/process_sequence_test.py`
  validates deterministic fp64 fixture schemas/repeatability, snapshots and
  allowed-field ownership mutation rules, independent inventory, dilution,
  wall-loss, slot, and exhaustion expectations, local malformed/alias rejection,
  and optional runtime Warp container/sidecar mirrors. It does not execute a
  direct process step or a process sequence.
- **P2 (complete, #1447):** The same module executes the five existing direct
  processes on shared test-local device state derived with all-enabled
  partitioning. Warp CPU is the required installed-Warp baseline; CUDA is
  optional and skips cleanly. Coverage includes final-only conversion guarding,
  condensation/nucleation inventory accounting, coagulation charge/mass and
  collision bounds, dilution and wall-loss budgets, no-ops, preflight
  immutability, exhaustion policy behavior, and persistent sidecar/RNG identity.
  Neutral wall-loss evidence is a separate stochastic aggregate test.
- **P3 (complete, #1448):**
  `particula/gpu/tests/gpu_complete_process_sequence_example_test.py` validates
  fixture schemas; forced and natural no-Warp paths; lazy loading; subprocess
  output; exactly one conversion per CPU container; the five-call order;
  caller-owned sidecar/RNG and resident-container identities; one synchronization
  followed by one `sync=False` restore per container; and real Warp CPU output.
  Parameterized loader, conversion, direct-step, synchronization, restore, and
  invalid-direct-input failures assert visible propagation with no fallback or
  premature checkpoint.
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
pytest particula/gpu/tests/gpu_complete_process_sequence_example_test.py -q -Werror \
  --cov=docs.Examples.gpu_complete_process_sequence --cov-report=term-missing \
  --cov-fail-under=80
pytest particula/gpu/tests/process_sequence_test.py -q -m "warp and gpu_parity and not cuda" -Werror
pytest particula/gpu/tests/process_sequence_test.py -q -m "warp and stochastic and not cuda" -Werror
pytest particula/gpu/tests/process_sequence_test.py -q -m "warp and cuda" -Werror
```
