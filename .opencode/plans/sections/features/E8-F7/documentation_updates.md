# Documentation Updates

## P5 Delivered

Issue #1593 (commit `dd3b0b55f`) published the canonical unavailable profiling
record at `docs/Features/gpu_graph_capture_performance.md`. It documents the
frozen small/medium matrix, exact reproduction commands, qualified native-CUDA
boundary, provenance/manifest/checksum rules, host-launch versus synchronized
elapsed versus Nsight methodology, and safety limits. No reviewed normalized
artifact or manifest is checked in; therefore no CUDA result, ranking, or
recommendation was published.

- Updated `docs/Features/Roadmap/data-oriented-gpu.md` and `AGENTS.md` to make
  E8-F7/T7 the profiling and recommendation owner and limit E8-F8 to the
  example, limitations, and closeout.
- Added `particula/tests/gpu_graph_capture_performance_docs_test.py`, a
  hardware-free cross-document contract for ownership, canonical content,
  commands, provenance, unavailable status, and no-fallback limits.
- Validated the document contract with `--no-cov` and rendering with
  `mkdocs build --strict`. The record intentionally contains no measured
  machine/software table, raw sample reference, bottleneck table, or
  recommendation until reviewed CUDA evidence exists.
