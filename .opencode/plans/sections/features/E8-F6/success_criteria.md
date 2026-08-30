# Success Criteria

- [ ] The opt-in CUDA matrix includes 1, 10, 100, and 1000 boxes where memory
  permits, with particles per box, species, active fraction, process set,
  communication, and diagnostics represented as explicit case metadata.
- [ ] Every infeasible or unavailable required row records a structured reason;
  no row silently disappears, falls back to CPU, or is reported as measured.
- [ ] Small and medium repeated-step workloads record uncaptured and captured raw
  samples from identical validated fixtures after separate warmup.
- [ ] Timing evidence records setup/capture separately from replay, explicit
  synchronization, timer, samples, summary statistics, command, versions,
  device, dimensions, seed, and UTC timestamps.
- [ ] The memory model accounts separately for primary state, inactive fixed
  slots, E8-F3 reusable resources, diagnostics, communication, checkpoints, and
  projected autodiff tape without double counting.
- [ ] Full-retention and checkpoint-interval tape scenarios use documented
  symbolic inputs and formulas and are never labeled measured or complete Epic I
  tape storage.
- [ ] Analytical logical bytes are compared with observed peak device-memory
  deltas for representative executed rows only when a documented allocator API
  is version-qualified with sufficient coverage; otherwise observed peaks are
  explicitly unavailable. Probe method, coverage, and unexplained delta are
  retained and allocator equality is not claimed.
- [ ] Sizing is deterministic and overflow-safe, and identical case/resource
  inputs produce byte-identical normalized evidence aside from timestamps and
  raw measured values.
- [ ] Default pytest collection remains unchanged; heavy rows require
  `--benchmark`, optional CUDA cleanly skips, and no performance number gates CI.
- [ ] Focused tests, untargeted repository coverage, linters, documentation
  contract tests, and `mkdocs build --strict` pass without lowering thresholds.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Required box-count rows represented | Direct-kernel matrix; no captured resident matrix | 1, 10, 100, 1000 executed or explicitly unavailable | Versioned benchmark artifact |
| Launch modes per comparable case | Uncaptured direct timings | Captured and uncaptured resident replay | Raw sample records |
| Raw timing provenance fields | Partial existing metadata | 100% required fields | Artifact schema tests |
| Resident memory categories | Direct benchmark estimates and E8-F3 reusable bytes | All seven named categories, no duplicate roles | Memory reconciliation tests |
| Representative analytical/observed comparisons | None for captured full loop | At least one small and one medium executed row when CUDA probe permits | Memory evidence artifact |
| Hidden CPU fallback for CUDA rows | Not allowed | 0 | Dispatch spies and device metadata |
| Full-package coverage threshold | Repository configured | Met or exceeded | `.opencode/tools/run_pytest.py` |

Speedup values are descriptive machine-specific evidence, not pass/fail targets.
