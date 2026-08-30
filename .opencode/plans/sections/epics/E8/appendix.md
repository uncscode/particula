# Appendix

## Source Material

- `docs/Features/Roadmap/data-oriented-gpu.md:1683` -- authoritative Epic H
  vision, eight planned feature tracks, exit bar, capture constraints, and
  performance/memory axes.
- `docs/Features/data-containers-and-gpu-foundations.md` -- container,
  transfer, fixed-slot, and resident ownership contracts.
- `particula/execution/resident_scheduler.py` -- shipped uncaptured full-loop
  ordering and failure boundary.
- `particula/execution/gpu_resources.py` -- stable resource and sidecar
  ownership boundary.
- `particula/execution/checkpoint.py` -- exact-device checkpoint/restart and
  continuation authority.
- `particula/gpu/kernels/tests/condensation_graph_capture_test.py` -- existing
  bounded direct-kernel graph-capture test patterns and clean-skip vocabulary.
- `particula/gpu/tests/benchmark_test.py` -- existing opt-in benchmark policy.

## Recapture Trigger Baseline

The first implementation must treat changes to any of the following as an
explicit teardown-and-recapture event unless a child plan proves a narrower
safe contract: device, backend, `n_boxes`, `n_particles`, `n_species`, array
dtype or identity, process order or enablement, communication kind/map,
diagnostic outputs, resource registry binding, RNG stream configuration,
checkpoint restore identity, or session lifecycle state.

## Alternatives Considered

- **A separate graph-only scheduler:** Rejected because it would duplicate
  process ordering and invite semantic drift from uncaptured execution.
- **Implicit capture on the first normal scheduler call:** Rejected because it
  hides allocation, device capability, and lifecycle transitions.
- **Automatic recapture on incompatibility:** Deferred because it obscures cost
  and mutation boundaries and complicates deterministic failure handling.
- **Benchmarks in default CI:** Rejected because graph capture and profiling
  require qualified CUDA hardware and would make correctness collection
  environment-dependent.
- **Dynamic resizing during replay:** Rejected; fixed-capacity inactive slots
  and shipped exhaustion policies are the compatible mechanism.

## Drafter Notes

- Classifier diagnostics: none.
- Research delegation was attempted but blocked by the subagent depth limit;
  the draft therefore used workflow messages, repository guidance, and the
  authoritative roadmap section directly.
- Epic plan records in the current plan schema do not support phase records;
  the attempted `add-phase` operation failed closed. The eight ordered feature
  child tracks serve as the epic workstream decomposition.
