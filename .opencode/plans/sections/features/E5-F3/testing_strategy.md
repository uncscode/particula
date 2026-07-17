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
- **P2 — Charged-only unit/integration tests:** Validate capability resolution
  and pre-mutation failures. Across fixed seeds, assert accepted pair invariants,
  bounded collision-count statistics against an independent expected sum, one
  RNG stream, caller-buffer identity, and separate per-box species-mass and
  total-charge conservation. Verify donor mass/concentration/charge clearing.
- **P3 — Combined integration/regression tests:** Assert candidate rates equal
  Brownian plus charged terms and the total bound covers each sum. Use repeated
  fresh seeded runs with declared aggregate/sigma bounds, not exact CPU/Warp pair
  replay. Prove canonical mechanism order is equivalent, only one pair buffer
  and RNG pass are used, persistent RNG advances without hidden reseeding, and
  legacy omitted/explicit Brownian results remain compatible.
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
