# Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
|------|--------|------------|------------|-------|
| Profiler instrumentation perturbs timing. | Kernel and launch conclusions become misleading. | High | Keep unprofiled benchmark timing separate from profiler runs; label methods and never mix samples. | E8-F7 implementer |
| Host launch and asynchronous device work are conflated. | Graph-capture benefit is over- or understated. | High | Record enqueue and synchronized elapsed categories separately with tested synchronization boundaries. | E8-F7 implementer |
| Kernel names or export schemas vary by Warp/Nsight version. | Parser breaks or misattributes work. | Medium | Version-gate parsers, retain raw exports, map only evidenced names, and preserve `unattributed`. | Profiling maintainer |
| Missing counters or permissions are treated as zero. | False bottlenecks or false confidence. | Medium | Represent unavailable metrics explicitly and block required closeout rows. | Profiling operator |
| One GPU result is generalized to all machines. | Users make invalid capacity/performance decisions. | High | Require device/software/workload bounds in every recommendation and documentation contract test. | Documentation owner |
| Workload state diverges between modes. | Captured/uncaptured comparison is invalid. | Medium | Reuse one qualified fixture per workload; drain, restore snapshot primary/continuation/RNG state without identity replacement, then validate identities before each batch. Reset incapability publishes unavailable evidence rather than rebuilding or recapturing. | Benchmark owner |
| Optimization advice changes scientific behavior. | Correctness and reproducibility regress. | Low | Reject contract-changing recommendations; create separate correctness-gated follow-up plans. | Epic H maintainers |
| Large or sensitive profiler artifacts are committed. | Repository bloat or metadata disclosure. | Medium | Commit compact summaries only; allow-list metadata and reference controlled local raw files by checksum. | E8-F7 implementer |
| Parent plan maps profiling to E8-F8 while orchestrator maps it to E8-F7. | Duplicate or missing closeout work. | High | Resolve parent metadata before issue generation and record one authoritative handoff. | Plan owner |
