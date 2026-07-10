# Open Questions

Status: reviewed and updated on 2026-07-10.

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
  single-box runtimes continue to climb across `1x500` through `1x50k`, while
  equivalent or larger total-particle multi-box cases scale much better across
  independent boxes.
- If a follow-up is needed, start with a staged design investigation. Compare
  pair-selection parallelism, collision-application parallelism, and graph
  capture before committing to a kernel rewrite.
