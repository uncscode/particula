# Dependencies

## Upstream

- **E8-F1:** capture capability, compatibility signature, lifecycle, explicit
  invalidation, recapture eligibility, and teardown vocabulary.
- **E8-F2:** capture-safe prepared resident enqueue boundary with host setup
  excluded from the captured path.
- **E8-F3:** complete pinned reusable resource set, identity report, and logical
  byte accounting.
- **E8-F4:** native complete-timestep capture, guarded replay, cleanup, and
  writer-failure behavior.
- **E8-F5:** CPU/uncaptured/captured correctness, conservation, communication,
  diagnostics, RNG, and recapture-trigger validation.
- **E8-F6:** opt-in multi-box timing and memory-budget evidence.
- **E8-F7:** CUDA profiling, bottleneck analysis, machine metadata, and bounded
  recommendations.
- Existing E7 resident example, testing policy, roadmap, MkDocs configuration,
  and plan validation tooling.

## Downstream

- Epic H status promotion and its handoff to Epic I depend on the complete,
  passing E8-F8 closeout record.
- Operators and future graph-capable workflows depend on the runbook's
  recapture and incident-response contract.
- Future performance work may consume the evidence, but must preserve its
  hardware/workload/version bounds.

## Phase Ordering

P1 establishes the executable path before P2 documents operations. P3 consumes
P1/P2 and all upstream evidence, and must derive the executable coverage target
list before running the command matrix. P4 is last and may mark Epic H Shipped
only when every required P3 row passes; otherwise it publishes blockers and
leaves the roadmap and plan Active.
