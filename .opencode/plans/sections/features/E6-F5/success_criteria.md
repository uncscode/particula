# Success Criteria

- [x] CPU and Warp discovery use the same active/free/invalid truth table.
- [x] Discovery free indices are ascending per box and unused fixed-shape entries are
  exactly `-1`.
- [x] Discovery active and free diagnostics equal independent NumPy oracle
  counts exactly for every tested box.
- [x] Request rank maps to the same slot on CPU, Warp CPU, and optional CUDA.
- [x] Activation preserves container and array identities, shapes, dtypes,
  devices, density, volume, request arrays, and every unselected slot.
- [x] Zero-request activation is an exact particle no-op with zero activated
  counts.
- [x] Contradictory state, malformed requests/sidecars, and insufficient
  capacity fail before any particle or caller-owned diagnostic mutation.
- [x] Development documentation records the authoritative CPU/direct-Warp
  contract, focused commands, and downstream ownership without adding a user
  example.
- [ ] E6-F6/F7/F8 can consume the documented discovery and activation contract
  without dynamic allocation or hidden transfer.
- [ ] Focused tests, full regressions, lint, type checking, and documentation
  validation pass without reducing coverage.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| CPU/Warp predicate disagreements | Undefined | 0 | Slot parity tests |
| Diagnostic count/index mismatches | No shared API | 0 | Independent NumPy oracle |
| Array shape or identity changes | Not supported | 0 | Identity regression tests |
| Invalid calls with observable writes | Not supported | 0 | Preflight snapshot tests |
| Dynamic allocations/resizes in activation kernel | N/A | 0 | Code review and fixed-shape tests |
| Changed-code coverage | Repository threshold | At least 80% | pytest-cov |
