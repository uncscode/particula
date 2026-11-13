# Documentation Updates

## Required Updates

- `docs/Features/Roadmap/data-oriented-gpu.md`
  - The Epic C planned-features text now records the one-thread-per-box path as
    accepted with caveat for many-box and direct-kernel use.
  - The roadmap status text now cross-references the existing measured decision
    record instead of leaving the acceptance decision implied.
  - The wording stays machine-bounded to the recorded evidence and does not
    broaden into a universal production guarantee.

- `docs/Features/data-containers-and-gpu-foundations.md`
  - The shipped support-boundary table now names the low-level GPU coagulation
    direct-kernel path as accepted with caveats.
  - Guidance bullets now state the appropriate uses (many independent boxes,
    Warp-backed direct-kernel workflows, CUDA-backed benchmark/study runs) and
    the caveated uses (large single-box production workloads, hidden-transfer
    expectations, and unsupported no-Warp environments).

## Optional Updates

- Add a follow-up issue/plan reference only if later evidence requires
  parallel-within-box work.

## Documentation Quality Requirements

- State CUDA optionality clearly.
- Avoid presenting machine-specific benchmark numbers as universal guarantees.
- Avoid implying hidden CPU/GPU synchronization or automatic transfers.
- Keep docs changes limited to roadmap/foundations guidance unless a paired
  benchmark source truly needs alignment.
- Keep future optimization work outside this feature's scope.
