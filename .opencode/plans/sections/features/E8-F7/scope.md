# Scope

E8-F7 adds an opt-in, CUDA-only profiling workflow around the final captured
and uncaptured resident timestep. It consumes upstream correctness, timing, and
memory artifacts; emits a stable machine-readable profile; and publishes a
machine-bounded interpretation. The orchestrator assigns this plan to T7
profiling even though older E8 parent text labels profiling as E8-F8; this plan
follows the explicit assignment and tracks reconciliation as an open question.

## In Scope

- Define small and medium representative resident workloads using fixed boxes,
  particles, species, process order, communication mode, and replay counts.
- Record device identity, architecture, driver/runtime/Warp/Python versions,
  command, warmup, sample count, synchronization method, and profiler versions.
- Measure uncaptured host dispatch, captured graph launch, synchronized
  end-to-end elapsed time, and derived launch-overhead contribution separately.
- Collect per-kernel duration, invocation count, occupancy-related metrics, and
  memory-throughput/access evidence with supported Warp/NVIDIA tooling.
- Rank bottlenecks by measured contribution and publish bounded, evidence-linked
  recommendations or explicit no-change decisions.
- Preserve raw profiler exports and normalized JSON summaries beneath the
  controlled benchmark artifact root; mark unavailable evidence explicitly.
- Add fast schema/analysis tests and CUDA-gated opt-in integration coverage.

## Out of Scope

- Changing scientific equations, process order, tolerances, RNG semantics, or
  ownership merely to improve a profiler metric.
- Promising performance portability across GPU models, drivers, Warp versions,
  workloads, box counts, or particle distributions.
- Adding a public profiler API, runtime auto-tuner, automatic kernel selection,
  hidden synchronization, CPU fallback, or default-CI benchmark execution.
- Replacing E8-F6 scaling and memory-budget evidence or re-measuring setup and
  graph-capture construction as replay cost.
- Implementing recommendations whose production changes belong in follow-up
  issues; this track identifies and bounds them.
