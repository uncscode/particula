# E3-F3 Scope

E3-F3 records the measured scaling limit of the current one-thread-per-box
implementation and updates docs to state that the shipped Epic C outcome is an
accepted-with-caveat boundary for many-box and low-level direct-kernel use.

## In Scope

- Record the accepted-with-caveat Epic C decision in
  `docs/Features/Roadmap/data-oriented-gpu.md` with a cross-reference to the
  shipped measured decision record.
- Update `docs/Features/data-containers-and-gpu-foundations.md` so the user-
  facing support table and guidance bullets state when the current low-level GPU
  coagulation path is appropriate and when it is caveated.
- Preserve CUDA and Warp optionality wording and the explicit CPU↔GPU transfer
  contract already documented in the public guides.
- Keep the shipped guidance bounded to many independent boxes, Warp-backed
  direct-kernel workflows, and CUDA-backed benchmark/study use.
- Document that large single-box production workloads remain caveated and do not
  become implied production recommendations.

## Out of Scope

- Production graph capture implementation or optimization.
- Rewriting the coagulation kernel for parallel pair selection within a box.
- Re-running or broadening benchmark code when the docs change can be anchored
  to the existing measured decision record.
- Broadening notebook-backed benchmark analysis beyond deterministic artifact
  lookup and source-of-record alignment.
- Changing CPU/GPU container transfer contracts or hidden synchronization
  behavior.
- Replacing E3-F1 RNG work or E3-F2 mixed-scale sampling decisions.
- Making CUDA mandatory in normal development, CI, or documentation builds.
