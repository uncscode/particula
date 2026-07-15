# GPU Condensation Support Contract, Examples, and Troubleshooting

## Problem Statement

Issues #1314 through #1316 completed the first three publication phases. The canonical foundations page states the bounded low-level GPU direct-condensation contract, the quick-start demonstrates its explicit-transfer workflow, and the troubleshooting section supplies reproducible failure guidance and focused commands. The published material prevents users from inferring unsupported high-level integration, hidden host transfers, or behavior beyond the fixed four-substep direct-kernel path.

## Value Proposition

The completed phases preserve explicit CPU-to-Warp transfer ownership, lazy direct/concrete imports, fixed-shape fp64 caller-owned sidecars, gas coupling, and explicit final checkpoint restoration. The published command matrix makes Warp `device="cpu"` the required baseline, treats CUDA as optional/local additive evidence, and keeps parity, inventory conservation, and energy bookkeeping as separate evidence classes. Text-only regressions protect the troubleshooting, command, migration-link, and README-discovery contracts without importing Warp or CUDA.

## User Stories

- As a scientific user, I want a precise GPU condensation support matrix so that I can select only verified vapor-pressure, activity, surface-tension, latent-heat, and gas-coupling modes.
- As an adopter, I want a runnable explicit-transfer example so that I can allocate reusable buffers and checkpoint CPU state without accidental per-step host synchronization.
- As a maintainer, I want focused commands and documentation guardrail tests so that published claims remain aligned with E4-F1 through E4-F6 evidence.

Parent epic: [E4](../../epics/E4/vision_problem.md). This is the final publication track after E4-F1 through E4-F6.
