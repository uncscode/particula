# Dependencies

## Upstream

- **E8-F1:** Supplies capture capability, lifecycle, compatibility signatures,
  invalidation, and recapture boundaries.
- **E8-F2:** Supplies allocation-free prepared enqueue paths and the closed
  requirement description used by benchmark setup.
- **E8-F3:** Supplies exact reusable resource identities and deterministic
  logical-byte accounting; P4 must consume this report.
- **E8-F4:** Supplies complete fixed-sequence graph capture and guarded replay.
- **E8-F5:** Supplies scientifically validated CPU/uncaptured/captured fixtures;
  benchmark rows may time only configurations that pass this correctness gate.
- Existing `--benchmark` gating, CUDA availability helpers, and artifact writer
  are infrastructure dependencies, not new public APIs.

## Downstream

- **E8-F7** consumes raw timings, summaries, memory observations, and unavailable
  rows for CUDA profiling.
- **E8-F8** uses measured setup/replay limitations and memory-budget examples in
  the runnable graph-capture documentation, runbook, and Epic H closeout.
- **Epic I** consumes the labeled tape-memory projections and later replaces
  them with measured differentiable-loop evidence.

## Phase Ordering

P1 defines records before P2/P3 emit benchmark rows. P2 establishes fair timing
before P3 expands the matrix. P4 depends on E8-F3 and may proceed after P1 in
parallel with P2/P3. P5 requires P2-P4 so observations use finalized cases and
categories. P6 publishes only executed or explicitly unavailable evidence after
all required validation commands are recorded.
