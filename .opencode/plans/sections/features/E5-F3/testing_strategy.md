# Testing Strategy

All code phases ship tests in
`particula/gpu/kernels/tests/coagulation_test.py`; scalar charged helper parity
remains owned by E5-F2 in
`particula/gpu/dynamics/tests/coagulation_funcs_test.py`. Tests use independent
NumPy/CPU calculations for expected physics rather than calling the Warp helper
under test as the oracle.

## Per-Phase Approach

- **P1 — Majorant unit tests (completed, issue #1342):** Co-located independent
  deterministic tests cover neutral, same-sign, opposite-sign, equal-size, and
  mixed nanometer/droplet physics fixtures; zero and invalid candidates; sparse
  compact active lists; and nonuniform per-box inputs. Direct-helper and
  dispatcher probes are compared with a local NumPy pair enumeration, including
  charged-dispatch addition and Brownian-only regression. These tests do not
  invoke public charged execution.
- **P2 — Charged-only unit/integration tests (completed, issue #1343):** Exact
  charged-only capability and forced finite-charge preflight are tested before
  caller-output/RNG mutation. Independent NumPy rate and compact-majorant probes
  cover signs, scales, sparse lists, nonuniform boxes, and multi-species summed
  total mass. Deterministic tests cover selector invariants, capacity, ownership,
  mass/charge conservation, and donor clearing; fixed fresh-seed aggregate tests
  cover neutral, same-sign, opposite-sign, and mixed-scale behavior.
- **P3 — Combined integration/regression tests (completed, issue #1344):**
  Independent NumPy fp64 probes assert candidate rates equal Brownian plus
  charged terms and that the one exhaustive additive-rate majorant covers each
  active-pair sum. Fresh seeded aggregate tests use declared bounds rather than
  exact CPU/Warp pair replay. Regression coverage proves canonical-order
  equivalence, exactly one selector and apply launch, one pair-buffer/RNG pass,
  persistent RNG reuse/reset behavior, Brownian compatibility, preflight state
  preservation, and tight per-box/per-species mass plus signed-charge
  conservation.
- **P4 — Documentation validation:** Check Markdown links, import paths,
  signature references, mechanism names, and any executable snippets.

## Device and Edge Coverage

- Warp CPU is required when Warp is installed; CUDA uses existing device
  parameterization and skips cleanly when unavailable.
- Include one-box and multi-box, one/multiple species, zero/one/two/many active
  particles, mixed-sign and zero charge, mixed nanometer/droplet scales,
  zero-collision calls, capacity limits, and caller-owned output/RNG buffers.
- Snapshot masses, concentration, charge, collision pairs/counts, and RNG state
  for every host-side validation failure.
- Use explicit `np.float64`, documented `rtol`/`atol` for deterministic values,
  and aggregate or sigma-based stochastic bounds.

## Coverage Policy

Coverage thresholds must never be lowered. Each changed function requires
co-located tests in the same phase/PR, test files retain the `*_test.py` suffix,
and changed code must meet at least 80% coverage. Focused GPU tests complement,
not replace, the existing full regression suite.
