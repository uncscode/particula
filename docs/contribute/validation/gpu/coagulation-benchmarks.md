# Historical coagulation benchmark evidence

This is the 2026-07-10 machine-bounded decision record preserved from revision
`7b064295037c3c9a137f5dd13c03f4a40da3f11b`. Its timings are historical observations, not measurements
of the v0.3.0 tree. The mutable local artifact path may now contain a different
run; inspect its metadata before comparison. It is not resident/capture evidence.

The repository does not retain generated files under `.artifacts/benchmarks/`.
Optional benchmark runs may recreate that ignored output directory. The
documentation viewer uses the declared snapshot in
`docs/data/gpu_benchmark_results.json`; its recorded provenance is unchanged
and does not imply that it contains the July capture described below.


- Command evidence: run
  `pytest particula/gpu/tests/benchmark_test.py --benchmark -v -s`.
  This remains opt-in benchmark evidence for local CUDA-capable environments,
  not part of the standard Warp CPU release-validation command set.
  The historical source-of-record artifact at
  `.artifacts/benchmarks/gpu_benchmark_results.json` preserves the exact
  executed local command in `benchmark_metadata.command`; the 2026-07-10 UTC
  capture came from a local coverage-wrapped invocation of that same benchmark
  target, but the plain user-facing reproduction command remains the one above.
- Hardware/context: the same artifact records `warp_version=1.15.0`,
  `device.alias=cuda:0`, `device.name=NVIDIA GeForce RTX 5060`, and
  `device.total_memory_bytes=8082096128` (about 8.08 GiB), with
  `started_at=2026-07-10T01:59:34.215800+00:00` and
  `updated_at=2026-07-10T02:00:43.918911+00:00`.
- Artifact output: `.artifacts/benchmarks/gpu_benchmark_results.json`
- Mixed-scale fixture note: the coagulation benchmark path uses a
  dedicated deterministic mixed NPF/droplet fixture aligned with the
  shipped coagulation rejection sampling baseline, while condensation benchmarks keep the generic
  helper.

## Recorded one-thread-per-box tradeoff

The recorded decision accepts the one-thread-per-box coagulation design with
caveats: it is the current documented baseline for many independent boxes,
especially when CUDA can deliver box-level parallel throughput, and for
low-level direct-kernel workflows, but large single-box use remains bounded
by the caution band below and does not imply a broader long-term scaling
decision.

- **Single-box caution band:** treat the recorded `1x10k` to `1x50k`
  range as the current caution band for large single-box workloads on the
  shipped one-thread-per-box kernel. The measured GPU time per step rises
  from `0.6809s` at `1x10k` to `1.3689s` at `1x20k` and `3.3984s` at
  `1x50k`, so this path is no longer a strong recommendation for large
  single-box work in that band. Smaller `1x500` to `1x5k` cases
  (`0.0361s`, `0.1366s`, `0.3411s`) remain a modest-throughput regime,
  not a decisive GPU win.
- **Multi-box effective region:** independent-box parallelism remains
  effective in the recorded many-box cases, where equivalent or larger
  total-particle workloads stay near the smaller timings seen at lower box
  counts: `10x500` `0.0359s`, `10x1k` `0.0699s`, `50x1k` `0.0714s`,
  `10x5k` `0.3437s`, `50x5k` `0.3528s`, `100x1k` `0.0766s`, and
  `10x10k` `0.6938s`. Record this as the region where the current GPU path
  still benefits from many independent boxes keeping threads busy.
- **Machine-bounded caveat:** this is measured evidence from the
  2026-07-10 capture on the recorded Warp/device context above, not a
  universal cutoff for every GPU or every future kernel variant. Treat it
  as the current decision record for the shipped one-thread-per-box design,
  not as a final acceptance decision on the long-term scaling strategy.


The [legacy viewer](legacy-benchmarks/gpu_benchmarks.ipynb) reads a locally
recorded artifact. [Resident benchmark qualification](resident-benchmarks.md)
requires its separate schema and reviewed evidence.
