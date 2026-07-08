# E3-F2 Open Questions

Status: reviewed and answered on 2026-07-08.

## Resolved Decisions

- Implement acceptance diagnostics as test-only helpers first. Only promote
  diagnostics to private optional buffers if tests show helper kernels cannot
  observe the needed acceptance behavior without distorting production kernels.
- Use the existing mass-precision scale anchors for mixed fixtures: NPF-scale
  particles near `1.5e-9 m` and droplet-scale particles near `8e-6 m` to
  `1.5e-5 m`. Keep particle counts small enough for deterministic fast tests,
  then use `benchmark_test.py --benchmark` for larger evidence.
- Keep fixed-bin prototypes bounded and measured. Start with a small bin count
  such as 4 or 8 and stop increasing once launch overhead or sequential
  one-thread-per-box work dominates acceptance gains.
- Define the final acceptance-rate threshold from E3-F2 measurements. The exit
  bar is either a measured improvement with conservation/statistical parity or
  an explicitly documented mixed-scale limitation with a caution boundary.
- Documentation should include statistical/conservation test evidence plus a
  short benchmark-style summary when CUDA benchmark data is available. Full
  benchmark execution remains opt-in.
