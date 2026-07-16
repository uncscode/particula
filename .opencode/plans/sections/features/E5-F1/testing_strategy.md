# Testing Strategy

Every implementation phase ships its tests in the same change. Coverage
thresholds remain unchanged and changed code must maintain at least 80%
coverage. Test files use the repository's `*_test.py` convention.

## Per-Phase Coverage

- **P1 — Configuration unit tests:** In
  `particula/gpu/kernels/tests/coagulation_test.py`, test omitted and explicit
  Brownian configurations; canonical ordering; empty, duplicate, unknown, and
  reserved terms; invalid distribution modes; stable error messages; and pure
  resolver behavior without requiring a launch. **Completed in Issue #1331:**
  the co-located host-only tests cover these resolver and capability cases;
  reserved-term failures name E5-F3, E5-F4, or E5-F5.
- **P2 — Dispatch and sampling tests (completed in Issue #1332):**
  `particula/gpu/kernels/tests/coagulation_test.py` covers private Brownian
  pair-rate/majorant helper parity against independent CPU formulas, sanitizer
  inputs, and reserved-mask no-op behavior. Test-only synthetic probes assert
  `K_total = sum(K_i)`, `M_total = sum(M_i)`, one candidate, and one acceptance
  draw. Fixed-seed same-backend regression, selector validity, and tight mass
  conservation preserve Brownian evidence; marked stochastic tests use
  aggregate independent-seed bounds. Invalid, zero, and underestimated inputs
  assert no accepted output or mutation.
- **P3 — Public integration/regression tests:** Under identical seeds, compare
  omitted configuration with explicit Brownian for collision counts, pairs,
  masses, concentration, and RNG state. For every rejected configuration,
  snapshot particle fields, caller collision buffers, count buffers, and
  persistent RNG and prove no value or object identity changes. Include
  one-box, multi-box, direct scalar, direct Warp-array, and environment-sidecar
  cases where relevant.
- **P4 — Documentation validation:** Check internal links, import paths, support
  table consistency, and executable snippets. Do not advertise reserved terms
  as available.

## Device and Stochastic Policy

- Warp CPU is required when Warp is installed; an unavailable Warp dependency
  follows existing collection-safe skips.
- CUDA is optional additive evidence and skips cleanly when unavailable.
- Pair-rate tests are deterministic with explicit `rtol`/`atol` per fixture.
- Sampling tests use seeded state for regression and aggregate or sigma-based
  bounds for stochastic behavior; exact CPU/Warp pair replay is not required.
- Tests independently check mass-preserving Brownian behavior; charge-transfer
  conservation remains E5-F2 scope.

## Verification Commands

```bash
pytest particula/gpu/kernels/tests/coagulation_test.py -q -k "mechanism or support" -Werror
pytest particula/gpu/kernels/tests/coagulation_test.py -q -Werror
ruff check particula/gpu/ --fix
ruff format particula/gpu/
ruff check particula/gpu/
mypy particula/gpu/ --ignore-missing-imports
```
