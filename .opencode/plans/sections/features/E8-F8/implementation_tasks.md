# Implementation Tasks

## Example and Operations Documentation

- [ ] Create `docs/Examples/gpu_resident_graph_capture.py` using upstream
  concrete-only setup, registry, prepared-plan, graph-capture, and teardown APIs.
- [ ] Keep Warp-dependent imports inside the qualified execution branch and
  report a precise unsupported result when CUDA capture is unavailable.
- [ ] Demonstrate explicit resident RNG initialization before capture, repeated
  replay without reseeding, caller synchronization before host inspection, and
  deterministic teardown.
- [ ] Demonstrate a structural incompatibility rejection and fresh explicit
  recapture without reusing the retired native handle.
- [ ] Write `docs/Features/gpu_graph_capture.md` with setup/replay/teardown
  procedures, ownership table, state machine, trigger matrix, failure recovery,
  clean-skip policy, limitations, and exact commands.

## Closeout and Roadmap

- [ ] Inventory executable modules changed by E8-F1--E8-F7 from implementation
  records and the final diff; freeze the complete production-module target list
  before running commands and exclude documentation-only and test files.
- [ ] Record date, commit, Warp availability, qualified devices, Python/Warp/
  driver/runtime versions, commands, literal output, and artifact checksums.
- [ ] Designate one qualified CUDA device for the closeout gate and require every
  required capture, correctness, benchmark, memory, and profiler row on that
  exact device and final revision. Keep additional device rows supplemental
  unless explicitly promoted to required.
- [ ] Link correctness, scaling, memory, launch-overhead, and profiling evidence
  to each E8 success metric; record unavailable evidence rather than omitting it.
- [ ] Accept only committed normalized E8-F6/E8-F7 artifacts whose schema,
  source revision, workload IDs, provenance, and SHA-256 match the closeout
  manifest; reject latest-path selection and manually copied summary values.
- [ ] Create `docs/Features/Roadmap/graph-capture-closeout.md` and leave status
  unshipped whenever any required row is failed, unavailable, or stale.
- [ ] Reconcile E8 parent child labels with the orchestrator assignment: E8-F7
  owns profiling and E8-F8 owns example/runbook/closeout.
- [ ] Update `data-oriented-gpu.md`, roadmap/docs indexes, `AGENTS.md`, and E8
  plan statuses with only stable, evidence-bounded conclusions.

## Tooling / Tests

- [ ] Add `particula/tests/gpu_resident_graph_capture_docs_test.py` for example
  sequence, concrete imports, lifecycle, limitations, and unsupported behavior.
- [ ] Add `particula/tests/gpu_graph_capture_closeout_docs_test.py` for report
  schema, required rows, exact commands, evidence links, and fail-closed status.
- [ ] Add CUDA-gated example execution coverage that never substitutes Warp CPU
  for capture evidence and cleanly skips only recognized capability absence.
- [ ] Run focused checks without coverage, then linters, the untargeted full
  coverage runner, plan validation, and `mkdocs build --strict` sequentially.
