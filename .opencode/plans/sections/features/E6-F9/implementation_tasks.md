# Implementation Tasks

### Validation

- [ ] Add deterministic one-box and multi-box fixtures to
  `particula/gpu/tests/process_sequence_test.py` with explicit fp64 inputs.
- [ ] Build independent expected budgets for dilution and wall loss and reuse
  CPU references from E6-F1/E6-F3/E6-F4 rather than GPU production helpers.
- [ ] Assert condensation and nucleation particle-plus-gas conservation per box
  and species, and coagulation mass/charge conservation.
- [ ] Assert slot activation/exhaustion counts and policy outcomes from E6-F5
  and E6-F6 without accepting silent demand truncation.
- [ ] Exercise persistent RNG across coagulation and stochastic wall loss
  without requiring identical CPU/Warp random streams.
- [ ] Snapshot all caller state for representative invalid calls and prove
  failure before particle, gas, sidecar, diagnostic, work-buffer, or RNG
  mutation.
- [ ] Mark Warp CPU as required when Warp is installed; add optional CUDA
  evidence with clean skip behavior.

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
