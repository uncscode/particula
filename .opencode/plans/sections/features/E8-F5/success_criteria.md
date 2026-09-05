# Success Criteria

- [x] One identical fixed-shape scenario runs for multiple timesteps through the
  CPU oracle and uncaptured Warp CPU path; every Warp-visible CUDA device runs
  captured qualification or records an independent unavailable result without
  fallback.
- [ ] The CUDA device designated for E8-F8 closeout completes the final E8-F5
  capture and correctness rows at the closeout revision; results from additional
  independently qualified devices are supplemental.
- [x] Primary particle, gas, and environment fields and every registered
  diagnostic are compared separately with explicit deterministic tolerances.
- [x] Per-box/per-species concentration-weighted inventory satisfies tight
  conservation independently of parity and stochastic assertions.
- [x] GAS and PARTICLES closed-map communication, optional volume evolution,
  empty/no-work rows, and stable resource identities are covered.
- [x] Coagulation and wall-loss RNG resources are nonaliasing, advance across
  work, preserve no-work rows, reset only explicitly, and continue through the
  supported same-device checkpoint/restart path.
- [x] Every documented structural or terminal lifecycle mismatch rejects before
  captured launch; post-launch writer failure faults the session and graph.
- [x] Replay performs no allocation, host transfer, bulk synchronization,
  validation scan, reseed, fallback, or automatic recapture.
- [x] Focused tests, untargeted repository coverage, and strict docs validation
  pass; optional unavailable hardware evidence remains visibly distinguished.

## P5 Gate Status (#1579)

- [x] Focused required evidence, optional CUDA clean-skip evidence, untargeted
  coverage, and hardware-free documentation assertions passed.
- [x] Strict MkDocs passed through the approved `docs-validator`
  `build_mkdocs_validate` worktree wrapper (strict mode is intrinsic). E8-F5 is
  shipped.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Execution paths compared | Existing resident regressions | CPU + uncaptured Warp + captured CUDA | `captured_full_loop_test.py` |
| Deterministic scenario length | Mostly one/two-step rows | At least 3 identical timesteps | Parity test parameters |
| Communication families | Separate resident tests | GAS and PARTICLES in full loop | Captured validation matrix |
| Preflight rejection launch count | No unified matrix | 0 launches for every invalid row | Capture-adapter spy |
| Conservation drift | Process-specific evidence | `rtol<=1e-12`, `atol<=1e-30` | Independent inventory oracle |
| Repository coverage threshold | Current configured threshold | Unchanged and passing | `.opencode/tools/run_pytest.py` |
