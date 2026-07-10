# Open Questions

Status: reviewed after shipped issue #1249 closeout updates.

## Resolved Decisions

- The refreshed P1 evidence now records the maintainer CUDA/Warp benchmark
  context directly in `docs/Features/Roadmap/data-oriented-gpu.md`: Warp 1.15.0
  on `cuda:0` (`NVIDIA GeForce RTX 5060`, 8 GiB; CUDA Toolkit 12.9,
  Driver 13.2), using
  `pytest particula/gpu/tests/benchmark_test.py --benchmark -v -s`.
- Raw benchmark output is stored at
  `.artifacts/benchmarks/gpu_benchmark_results.json`, while the roadmap prose
  keeps only the compact decision-relevant timing summary.
- E3-F2's bounded mixed-scale result is now reflected in the coagulation-only
  benchmark fixture: the opt-in coagulation path uses a dedicated mixed
  NPF/droplet helper and focused regression coverage confirms condensation did
  not inherit that fixture.
- The shipped timing note now makes the current caution boundary explicit:
  treat the recorded `1x10k` to `1x50k` range as the measured large single-box
  caution band for the current one-thread-per-box path, while smaller
  `1x500` to `1x5k` runs are only a modest-throughput regime rather than a
  strong GPU win.
- The delivered roadmap decision record also names the measured many-box
  effective region explicitly: `10x500`, `10x1k`, `50x1k`, `10x5k`, `50x5k`,
  `100x1k`, and `10x10k` remain the evidence-backed independent-box cases.
- The notebook-backed theory/example text now keeps the controlled artifact path
  as the preferred source of record and accepts fallback exports only when they
  carry matching benchmark provenance, so roadmap and example guidance no longer
  diverge on where the benchmark evidence lives or how broadly it should be
  interpreted.
- Issue #1248 resolved the remaining public-docs ambiguity without changing the
  kernel: the roadmap now marks the current one-thread-per-box path as accepted
  with caveat for Epic C, and the foundations guide now tells users when to use
  it and when not to.
- Issue #1249 closed the follow-up decision without opening a new track; the
  remaining closeout work in this branch is limited to benchmark helper,
  example-artifact, and plan-record consistency fixes.
- No parallel-within-box follow-up is currently scoped. If later evidence
  requires one, start
  with a staged design investigation that compares pair-selection parallelism,
  collision-application parallelism, and graph capture before committing to a
  kernel rewrite.
