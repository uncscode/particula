# Dependencies

## Upstream

- **E8 parent:** stable-shape fixed-order resident graph, explicit setup/replay/
  teardown, no hidden fallback, and three-way validation requirements.
- **E8-F1:** capture capability, compatibility signature, invalidation reasons,
  lifecycle state machine, and explicit recapture gate.
- **E8-F2:** immutable prepared resident plan and the shared uncaptured,
  capture-safe device enqueue sequence.
- **E8-F3:** complete pinned capture resource set, stable identities, and
  allocation/accounting contract.
- **E8-F4:** concrete graph owner, native capture/replay boundary, teardown, and
  post-launch fault behavior.
- **E7/E6 execution:** resident scheduler, diagnostics, communication,
  checkpoints, RNG streams, and fixed-capacity process seams.
- **External:** NumPy, pytest/pytest-cov, and Warp. CUDA hardware is optional but
  captured evidence must never substitute Warp CPU or silently fall back.

## Downstream / Sibling Features

- E8 scaling benchmarks must run only after this correctness matrix passes.
- E8 memory-budget work uses the validated state/resource inventory.
- E8 graph-capture examples and limitations cite these supported scenarios and
  clean-skip behavior.
- E8 profiling and closeout consume this command matrix and literal results.

## Phase Ordering

P1 fixes the shared oracle before either GPU path. P2 establishes the uncaptured
baseline. P3 compares capture against both prior paths. P4 extends the stable
fixture to RNG and rejection semantics. P5 records integrated evidence and docs
last. The parent child-plan table labels captured validation as E8-F4 while this
created record and orchestrator handoff assign it to E8-F5; implementation should
follow this plan ID unless orchestration metadata is corrected before issue split.
