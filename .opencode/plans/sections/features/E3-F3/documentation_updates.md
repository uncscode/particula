# Documentation Updates

## Required Updates

- `docs/Features/Roadmap/data-oriented-gpu.md`
  - Mark the one-thread-per-box coagulation decision as accepted, accepted with
    caveats, or requiring a follow-up.
  - Include measured single-box and multi-box scaling limits with command and
    hardware context.
- Reference E3-F2 if mixed-scale sampling results affect interpretation.
- Preserve the one-thread-per-box decision as evidence-driven: summarize the
  benchmark artifacts and measured limits that justify keeping, caveating, or
  replacing the current approach.

- `docs/Theory/nvidia-warp/examples/gpu_benchmarks.py`
  - Refresh benchmark interpretation text if existing single-box and multi-box
    claims are stale.
  - Keep this `.py` source authoritative for paired notebook updates.

- `docs/Features/data-containers-and-gpu-foundations.md` or nearby feature docs
  - Add concise guidance explaining where current low-level coagulation GPU APIs
    are appropriate and where future parallel work belongs.

## Optional Updates

- Add a small benchmark result artifact if maintainers prefer separating raw
  measurements from prose documentation.
- Add a follow-up issue/plan reference if parallel-within-box work is scoped.

## Documentation Quality Requirements

- State CUDA optionality clearly.
- Avoid presenting machine-specific benchmark numbers as universal guarantees.
- Avoid implying hidden CPU/GPU synchronization or automatic transfers.
- Keep future optimization work outside this feature's scope.
