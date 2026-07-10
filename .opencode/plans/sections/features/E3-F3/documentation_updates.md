# Documentation Updates

## Required Updates

- `docs/Features/Roadmap/data-oriented-gpu.md`
  - Added a shipped coagulation benchmark evidence block under
    `### Performance and Memory` with the exact pytest command, benchmark date,
    CUDA/Warp/device context, artifact path, and compact single-box vs
    multi-box timing summary.
  - Explicitly notes that the coagulation benchmark path now uses a dedicated
    deterministic mixed NPF/droplet fixture aligned with the shipped E3-F2
    baseline, while condensation benchmarks keep the generic helper.
  - Keeps the interpretation bounded to current evidence for the existing
    one-thread-per-box design rather than claiming the final long-term scaling
    decision is complete.

- Reference E3-F2 mixed-scale evidence when interpreting the refreshed
  coagulation benchmark fixture and timing note.

## Optional Updates

- Store raw benchmark output under `.artifacts/benchmarks/` when produced; the
  shipped roadmap note should summarize only the decision-relevant evidence.
- Add a follow-up issue/plan reference if parallel-within-box work is scoped.

## Documentation Quality Requirements

- State CUDA optionality clearly.
- Avoid presenting machine-specific benchmark numbers as universal guarantees.
- Avoid implying hidden CPU/GPU synchronization or automatic transfers.
- Keep future optimization work outside this feature's scope.
