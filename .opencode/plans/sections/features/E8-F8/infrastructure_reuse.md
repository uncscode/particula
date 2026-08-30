# Infrastructure Reuse

- `particula.execution.graph_capture` (introduced by E8-F1/E8-F4) -- reuse its
  concrete-only capability, compatibility signature, lifecycle, capture,
  replay, invalidation, and teardown operations; do not create an example-only
  wrapper or package export.
- `particula.execution.gpu_session` and `gpu_resources` -- reuse exact ACTIVE
  session, closed guard, pinned-resource identity, and ownership rules described
  by `.opencode/plans/sections/epics/E8/implementation_strategy.md:5-35`.
- `particula.execution.resident_scheduler` -- retain its fixed twelve-node
  sequence and writer-failure semantics; the example must not become a second
  scheduler.
- `particula/execution/process_graph.py:197-367` -- reuse the closed resident
  node catalogue and dependency ordering when explaining fixed process order.
- `particula/gpu/kernels/tests/condensation_graph_capture_test.py:186-253` --
  follow existing precise capability skips and one-time capture cleanup rather
  than swallowing arbitrary failures.
- `docs/Examples/gpu_resident_multi_timestep.py` -- follow the existing
  canonical resident setup, explicit transfer, resource, RNG, checkpoint, and
  synchronization conventions when creating the graph-capture example.
- `particula/tests/gpu_resident_multi_timestep_docs_test.py` -- follow the
  hardware-free AST/text contract-test pattern for imports, sequence,
  limitations, and executable documentation.
- `.opencode/guides/testing_guide.md:198-273` -- use Warp CPU for hardware-free
  contracts, optional CUDA pass-or-clean-skip evidence, focused coverage-free
  checks, and the untargeted repository runner for comprehensive coverage.
- `docs/Features/Roadmap/data-oriented-gpu.md:1683-1785` -- update the Epic H
  planned features, recapture rules, performance boundaries, and exit bar rather
  than adding a competing roadmap.
- E8-F5 correctness, E8-F6 scaling/memory, and E8-F7 profiling artifacts -- link
  their normalized evidence and exact reproduction commands; do not copy or
  reinterpret raw data.
