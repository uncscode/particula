# Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-09-06 | Shipped E8-F7-P1 for issue #1589: host-only strict profiling evidence records, exact small/medium frozen workload matrix, executed/unavailable union, bounded canonical JSON, and safe injected `.artifacts` raw provenance in `profiling_support.py` with hardware-free tests. No Warp/CUDA/profiler process, public export, or timing evidence was added. | implementation |
| 2026-08-30 | Selected ignored local-only raw Nsight retention under `.artifacts/benchmarks/profiling/raw/`; only bounded normalized summaries, raw checksums, and the explicit non-shared limitation are committed | user decision |
| 2026-08-30 | Required E8-F7 to freeze canonical small and medium workload IDs from the smallest launch-sensitive and largest repeatably feasible E8-F6 rows, with no per-device substitution | user decision |
| 2026-08-30 | Resolved the profiler export contract with the official Arch Linux Nsight Systems/Compute package pair, recorded the local RTX 5060/driver/CUDA executable identities, defined bounded Python subprocess and parser ownership plus an opt-in real-binary smoke test, set a fail-closed metric floor, and retained an NVIDIA CUDA GPU-only collection boundary | user decision |
| 2026-08-30 | Initial T7 plan for representative CUDA profiling, launch/kernel cost separation, bottleneck analysis, and machine-bounded recommendations; preserved classifier diagnostics (`none`) and recorded the E8-F7/E8-F8 mapping conflict | plan-feature-drafter |
