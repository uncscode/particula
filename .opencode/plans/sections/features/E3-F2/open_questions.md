# E3-F2 Open Questions

Status: reviewed and answered on 2026-07-08.

## Resolved Decisions

- Implement acceptance diagnostics as test-only helpers first. Only promote
  diagnostics to private optional buffers if tests show helper kernels cannot
  observe the needed acceptance behavior without distorting production kernels.
- Outcome: resolved in issue #1241 by keeping the mirrored attempt-count kernel
  and seeded comparison helper entirely inside
  `particula/gpu/kernels/tests/coagulation_test.py`; no production API or
  private exported diagnostic buffer was needed.
- Use the existing mass-precision scale anchors for mixed fixtures: NPF-scale
  particles near `1.5e-9 m` and droplet-scale particles near `8e-6 m` to
  `1.5e-5 m`. Keep particle counts small enough for deterministic fast tests,
  then use `benchmark_test.py --benchmark` for larger evidence.
- Keep fixed-bin prototypes bounded and measured. Start with a small bin count
  such as 4 or 8 and stop increasing once launch overhead or sequential
  one-thread-per-box work dominates acceptance gains.
- Outcome: issue #1242 did not ship a fixed-bin path. P2 answered the immediate
  hardening question with bounded active-particle rank selection inside the
  existing global-majorant sampler, which removed invalid retry proposals
  without adding bin-local structures or changing Brownian physics.
- Define the final acceptance-rate threshold from E3-F2 measurements. The exit
  bar is either a measured improvement with conservation/statistical parity or
  an explicitly documented mixed-scale limitation with a caution boundary.
- Status: still open for P3/P4. Issue #1242 added selector-validity,
  sparse/degenerate, exactly-two-active, bounds, and conservation coverage, but
  it did not yet establish the final mixed-scale acceptance threshold or final
  user-facing limitation language.
- Documentation should include statistical/conservation test evidence plus a
  short benchmark-style summary when CUDA benchmark data is available. Full
  benchmark execution remains opt-in.
