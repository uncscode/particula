# Implementation Tasks

### Validation

- [x] Add deterministic one-box and multi-box fixtures to
  `particula/gpu/tests/process_sequence_test.py` with explicit fp64 inputs.
- [x] Build independent expected inventory, dilution, wall-loss, slot, and CPU
  exhaustion-planner expectations without invoking GPU process physics helpers.
- [x] Snapshot caller-owned particle, gas, sidecar, diagnostic, work-buffer,
  and RNG state; test local invalid fixture/alias rejection before mutation.
- [x] Add an optional runtime Warp CPU/CUDA mirror that verifies fixture-backed
  container and stale-sidecar identity/schema without executing a process step.
- [x] Exercise all five existing direct GPU boundaries consecutively on private
  resident test state, with persistent same-device sidecars/RNGs, stable
  identities, final-only conversion, accounting/no-op/preflight/exhaustion
  checks, required Warp CPU coverage, and optional CUDA coverage. (P2, #1447)

### Example

- [ ] Add `docs/Examples/gpu_complete_process_sequence.py` with lazy Warp-only
  imports and a deterministic CPU-only no-kernel branch.
- [ ] Convert CPU containers once, allocate/reuse caller-owned sidecars, call
  all five direct processes, and restore only at the final checkpoint.
- [ ] Print stable process order, shape, diagnostic, transfer-boundary, and
  support-boundary summaries without claiming a general scheduler.
- [ ] Add `gpu_complete_process_sequence_example_test.py` for imports,
  subprocess output, identities, process order, explicit transfers, and errors.

### Documentation and Planning

- [ ] Update the relevant `docs/Features/` pages and `AGENTS.md` with direct
  entry points, ownership, diagnostics, focused tests, and unsupported scope.
- [ ] Add an E6/E6-F1-through-E6-F9 inventory and artifact links to both GPU
  roadmap documents.
- [ ] Reconcile the Epic F feature list and exit bar with the shipped contracts
  and integrated sequence evidence.
- [ ] Keep Epic G pending/on deck; explicitly assign backend selection,
  high-level runnables, process scheduling, and resident multi-step loops to it.
- [ ] Validate all plan records and mark E6 shipped only after every child phase
  and closeout check is actually complete.
