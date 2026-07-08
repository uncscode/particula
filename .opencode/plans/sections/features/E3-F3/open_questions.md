# Open Questions

Status: reviewed and answered on 2026-07-08.

## Resolved Decisions

- Treat the maintainer's CUDA/Warp development machine as the reference context
  for refreshed benchmark numbers, and record Warp, CUDA, device, driver, and
  Python versions beside the results.
- Store raw benchmark output as an artifact under `.artifacts/benchmarks/` and
  summarize only the decision-relevant numbers in roadmap prose.
- Final single-box scaling interpretation should wait until E3-F2 either
  hardens mixed-scale sampling or documents its bounded limitation.
- Pick the particle-count caution boundary from measured results, not before
  data collection. The current benchmark matrix already spans `1x10k`, `1x20k`,
  and `1x50k` coagulation GPU-only cases for this decision.
- If a follow-up is needed, start with a staged design investigation. Compare
  pair-selection parallelism, collision-application parallelism, and graph
  capture before committing to a kernel rewrite.
